#  ======================== MONTE CARLO TREE SEARCH ========================== #
# Project:          Symbolic regression tests
# Name:             EvaluateFit.py
# Description:      Tentative implementation of a basic MCTS
# Authors:          Vincent Reverdy & Jean-Philippe Bruneton
# Date:             2018
# License:          BSD 3-Clause License
# ============================================================================ #


# ================================= PREAMBLE ================================= #
# Packages
import numpy as np
import config
import copy
import cma
import matplotlib.pyplot as plt

import sys
# ============================ CLASS: Evaluate_Fit ================================ #
# A class returning the reward of a given equation w.r.t. the target data (or function)
class Evaluatefit:

    def __init__(self, formulas, voc, target, tolerance, mode):
        self.formulas = copy.deepcopy(formulas)
        self.tolerance = tolerance
        self.target = target
        self.voc = voc
        self.mode = mode
        self.scalar_numbers = 0
        self.n_targets, self.n_variables, self.variables, self.targets, self.f_renormalization, self.ranges, self.maximal_size = target.target
        self.xsize = self.variables[0].shape[0]
        self.step = (self.ranges[0])/self.xsize
        if self.n_variables > 1:
            self.ysize = self.variables[0].shape[1]
        if self.n_variables > 2:
            self.zsize = self.variables[0].shape[2]

    # ---------------------------------------------------------------------------- #
    def rename_formulas(self):
        ''' index all the scalar 'A' by a A1, A2, etc, rename properly the differentials, and finally resize as it must '''
        neweq = ''

        # rename the A's
        self.scalar_numbers = self.formulas.count('A')

        if self.scalar_numbers == 1:
            self.formulas += '+ A'
            self.scalar_numbers = 2

        A_count = 0
        for char in self.formulas:
            if char == 'A':
                neweq += 'A[' + str(A_count) + ']'
                A_count += 1
            else:
                neweq += char

        highest_der = 0
        for u in range(1,config.max_derivative):
            if 'd'*u in neweq:
                highest_der += 1

        if highest_der != 0 :

            for u in range(1, highest_der+1):
                if highest_der-u > 0:
                    arr = '[:-' + str(highest_der-u) + ']'
                else:
                    arr = '[:]'
                look_for = 'd'*u + '_x0'*u + '_f0'
                replace_by = 'np.diff(f[0]' + arr + ',' +str(u) +')/('  + str(self.step) +'**' +str(u)+')'
                neweq = neweq.replace(look_for, replace_by)

        if highest_der != 0 :
            base_array = '[:-' + str(highest_der) + ']'
        else:
            base_array = '[:]'
        string_to_replace = 'x0'
        replace_by = '(x[0]' + base_array +')'#+ '*' + str(self.ranges[0]) + ')'
        neweq = neweq.replace(string_to_replace, replace_by)

        string_to_replace = 'f0'
        replace_by = 'f[0]' + base_array
        neweq = neweq.replace(string_to_replace, replace_by)

        string_to_replace  = 'one'
        replace_by = '1.0'
        neweq = neweq.replace(string_to_replace, replace_by)

        string_to_replace = 'two'
        replace_by = '2.0'
        neweq = neweq.replace(string_to_replace, replace_by)

        string_to_replace = 'neutral'
        replace_by = '1.0'
        neweq = neweq.replace(string_to_replace, replace_by)

        string_to_replace = 'zero'
        replace_by = '0.0'
        neweq = neweq.replace(string_to_replace, replace_by)
        self.formulas = neweq

    # ---------------------------------------------------------------------------- #
    def reward_formula(self, error_on_target):
        u = error_on_target / self.tolerance

        if u >= 2:
            reward = -1
        else:
            reward = -np.arcsin(u - 1) * 2 / np.pi

        return reward

    # ---------------------------------------------------------------------------- #
    def formula_eval(self, x, f, A) :
        try:
            toreturn = eval(self.formulas)/self.f_renormalization
            if type(toreturn) != np.ndarray or np.isnan(np.sum(toreturn)) or np.isinf(np.sum(toreturn)) :
                return False, None
            else:
                return True, toreturn

        except (RuntimeWarning, RuntimeError, ValueError, ZeroDivisionError, OverflowError, SystemError, AttributeError):

            return False, None

    # ---------------------------------------------------------------------------- #
    def evaluation_target(self, a):
        err = 0
        success, eval = self.formula_eval(self.variables, self.targets, a)

        if success == True:
            mavraitarget = np.diff(self.targets[0], config.max_derivative) / (self.step ** (config.max_derivative))

            if config.trytrick:
                x0 = self.variables[0][:-2]
                #mavraitarget = mavraitarget * (0.04189318 + 1.39177054*np.cos(1.80152565*x0)-1.06877709*np.sin(1.20986685*x0) - 0.16845738*self.targets[0][:-2])*10**(-12)
                mavraitarget = mavraitarget +9.75024*self.targets[0][:-2]+2.22983*np.diff(self.targets[0], 1)-7.79562

            resize_eval = eval[:mavraitarget.size]
            diff = resize_eval - mavraitarget
            err += (np.sum(diff**2))
            err /= np.sum(np.ones_like(diff))
        else:
            return 1200000

        return err

    # -----------------------------------------
    def finish_with_least_squares_target(self, a):
        # this flattens the training data : this is required for least squares method : must be a size-1 array!
        flatfun = []
        res = self.func(a, self.targets, self.variables)

        if self.n_variables == 1:
            for i in range(self.xsize):
                flatfun.append(res[i])

        if self.n_variables == 2:
            for i in range(self.xsize):
                for j in range(self.ysize):
                    flatfun.append(res[i, j])

        if self.n_variables == 3:
            for i in range(self.xsize):
                for j in range(self.ysize):
                    for k in range(self.zsize):
                        flatfun.append(res[i, j, k])

        return np.asarray(flatfun)

    # ---------------------------------------------------------------------------- #
    def fit(self, reco):
        # applies least square fit starting from the recommendation of cmaes :
        x0 = reco
        return least_squares(self.finish_with_least_squares_target, x0, jac='2-point', loss='cauchy', args=())

    # ---------------------------------------------------------------------------- #
    def func(self, A, f, x):
        # eval the func, leaves only A undefined
        toeval = self.formulas + '-f*self.f_renormalization'
        return eval(toeval)

    # -------------------------------------------------------------------------------  #
    def best_A_cmaes(self):
        # applies the cmaes fit:
        initialguess = 2*np.random.rand(self.scalar_numbers)-1
        initialsigma = np.random.randint(1,5)

        try:
            res = cma.CMAEvolutionStrategy(initialguess, initialsigma,
                {'verb_disp': 0}).optimize(self.evaluation_target).result

            reco = res.xfavorite
            rec = []

            for u in range(reco.size):
                rec.append(reco[u])

        except (RuntimeWarning, RuntimeError, ValueError, ZeroDivisionError, OverflowError, SystemError, AttributeError):

            return False, [1]*self.scalar_numbers

        return True, rec

    # -------------------------------------------------------------------------------  #
    def best_A_least_squares(self, reco):
    # calls least square fit from cmaes reco :
        try:
            ls_attempt = self.fit(reco)

        except (RuntimeWarning, RuntimeError, ValueError, ZeroDivisionError, OverflowError, SystemError, AttributeError):
            return False, [1]*self.scalar_numbers

        success = ls_attempt.success
        if success:
            reco_ls = ls_attempt.x
            # transforms array into list
            rec = []
            for u in range(reco_ls.size):
                rec.append(reco_ls[u])
            return True, rec

        else:
            return False, [1]*self.scalar_numbers


    def eval_reward_nrmse(self, A):
    #for validation only

        success, result = self.formula_eval(self.variables, self.targets, A)

        if success:
            mavraitarget = np.diff(self.targets[0], config.max_derivative)/(self.step**(config.max_derivative))
            plt.plot(mavraitarget)
            plt.show()
            if config.trytrick:
                x0 = self.variables[0][:-2]
                #mavraitarget = mavraitarget * (0.04189318 + 1.39177054 * np.cos(1.80152565 * x0) - 1.06877709 * np.sin(
                #    1.20986685 * x0) - 0.16845738 * self.targets[0][:-2]) * 10 ** (-12)
                mavraitarget = mavraitarget +10*self.targets[0][:-2]+2*np.diff(self.targets[0], 1)[:-1]/self.step-8
                plt.plot(mavraitarget)
                plt.show()
            resize_result = result[:mavraitarget.size]
            plot = config.plot
            if plot:
                plt.plot(mavraitarget)
                plt.plot(result, 'r')
                plt.show()
            quadratic_cost = np.sum((mavraitarget - resize_result)**2)
            n = mavraitarget.size
            rmse = np.sqrt(quadratic_cost / n)
            nrmse = rmse / np.std(mavraitarget)
            return nrmse

        else:
            return 100000000
    # ---------------------------------------------------------------------------- #
    def eval_reward(self, A):
        derivative_cost = np.zeros(self.n_variables)
        reward = -1
        if self.mode == 'train':
            usederivativecost = 1
        else:
            usederivativecost = 0
        success, result = self.formula_eval(self.variables, self.targets, A)
        if success:
            mavraitarget = np.diff(self.targets[0], config.max_derivative)/(self.step**(config.max_derivative))
            resize_result = result[:mavraitarget.size]
            distance_cost = np.sum(np.absolute(mavraitarget - resize_result))

            if config.usederivativecost:
                for i in range(self.n_variables):
                    differential_along_i = np.diff(mavraitarget - resize_result, axis=i)
                    myvar = self.variables[i]
                    diff_my_var = np.diff(myvar, axis=i)
                    derivative_along_i = np.divide(differential_along_i, diff_my_var)
                    derivative_cost[i] = np.sum(np.absolute(derivative_along_i))
                error_on_target = distance_cost + usederivativecost *config.usederivativecost * np.sum(derivative_cost)
            else:
                error_on_target = distance_cost
            reward = self.reward_formula(error_on_target)

            # add parsimony cost
            if len(A) >= config.parsimony:
                A_cost = config.parsimony_cost * (len(A) - config.parsimony)
                reward -= A_cost
                if reward < -1:
                    reward = -1

        if np.isnan(reward) or np.isinf(reward):
            reward =-1

        return reward

    # ------------------------------------------------------------------------------- #
    def evaluate(self):
        ''' evaluate the reward of an equation'''

        np.seterr(all = 'ignore')
        allA = []
        failure_reward = -1
        self.rename_formulas()
        if self.scalar_numbers == 0:
            reward = self.eval_reward(allA)
            rms = self.eval_reward_nrmse(allA)
            if rms > 100000000:
                rms = 100000000
            return reward, self.scalar_numbers, allA, rms

        # else: cmaes fit : ---------------------------------------------------------- #
        success, allA = self.best_A_cmaes()
        if success == False:
            return failure_reward, self.scalar_numbers, [1]*self.scalar_numbers, 100000000

        # ---------------------------------------------------------------------------- #
        #else, compute some actual reward:
        reward_cmaes = self.eval_reward(allA)
        rms = self.eval_reward_nrmse(allA)

        #now compare the three and chose the best
        if rms > 100000000:
            rms = 100000000

        return reward_cmaes, self.scalar_numbers, allA, rms