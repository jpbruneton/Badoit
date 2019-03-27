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
from scipy.optimize import least_squares
import copy
import cma


# ============================ CLASS: Evaluate_Fit ================================ #
# A class returning the reward of a given equation w.r.t. the target data (or function)
class Evaluatefit:

    def __init__(self, formulas, voc, target, tolerance, mode):
        self.formulas = copy.deepcopy(formulas)
        self.tolerance = tolerance
        self.target = target
        self.voc = voc
        self.mode = mode

        self.n_targets, self.n_variables, self.variables, self.targets, self.f_renormalization, self.ranges, self.maximal_size = target.target
        self.xsize = self.variables[0].shape[0]
        if self.n_variables > 1:
            self.ysize = self.variables[0].shape[1]
        if self.n_variables > 2:
            self.zsize = self.variables[0].shape[2]

    # ---------------------------------------------------------------------------- #
    def rename_formulas(self):
        ''' index all the scalar 'A' by a A1, A2, etc, rename properly the differentials, and finally resize as it must '''

        neweq = ''
        # rename the A's
        scalar_numbers = self.formulas.count('A')
        A_count = 0
        for char in self.formulas:
            if char == 'A':
                neweq += 'A[' + str(A_count) + ']'
                A_count += 1
            else:
                neweq += char
        # rename by hand: this transforms x0 into x0[:] if one variable ... or into x0[:,:,:] if three variables
        arr = ''
        for j in range(self.n_variables):
            arr = arr + ':'
            if j < self.n_variables - 1:
                arr = arr + ','

        # renaming f: this transforms f into f[:] if one variable ... or into f[:,:,:] if three variables
        for i in range(self.n_targets):
            string_to_replace = 'f' + str(i)
            replace_by = 'f' + str(i) + '[' + arr + ']'
            neweq = neweq.replace(string_to_replace, replace_by)


        # rename the x : transforms x0 in x[0], x1 in x[1], x2 in x[2] (because variables are self.variables = [X,Y,Z], see class Targets)
        for i in range(self.n_variables):
            string_to_replace = 'x' + str(i)
            replace_by = '(' + 'x' + '[' + str(i) +']' + '[' + arr + ']' +'*' + str(self.ranges[i]) + ')'
            neweq = neweq.replace(string_to_replace, replace_by)

        # the update
        self.formulas = neweq

        return scalar_numbers


    # ---------------------------------------------------------------------------- #
    def reward_formula(self, error_on_target):
        u = error_on_target / self.tolerance

        if u >= 2:
            reward = -1
        else:
            reward = -np.arcsin(u - 1) * 2 / np.pi

        return reward

    # ---------------------------------------------------------------------------- #
    def formula_eval(self, x, A) :
        try:
            toreturn = eval(self.formulas)/self.f_renormalization
            if np.isnan(np.sum(toreturn)) or np.isinf(np.sum(toreturn)):
                return False, None
            else:
                return True, toreturn

        except (RuntimeWarning, RuntimeError, ValueError, ZeroDivisionError, OverflowError, SystemError, AttributeError):
            return False, None

    # ---------------------------------------------------------------------------- #
    def evaluation_target(self, a):
        err = 0

        success, eval = self.formula_eval(self.variables, a)

        if success == True:
            diff = eval - self.targets[0]
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
    def best_A_cmaes(self, scalar_numbers):
        #applies the cmaes fit:

        if scalar_numbers == 1:
            # cant use CMAES since in this case it returns : "ValueError: optimization in 1-D is not supported (code was never tested)"
            # then just make a trick : add one new scalar!
            self.formulas += '+A[1]'
            scalar_numbers = 2

        # then, randomize initial guess between -1 and 1, and initial sigma between 1 and 5:
        initialguess = 2*np.random.rand(scalar_numbers)-1
        initialsigma = np.random.randint(1,5)

        try:
            res = cma.CMAEvolutionStrategy(initialguess, initialsigma,
                {'verb_disp': 0, 'maxfevals' : '1e4 * N**2', 'popsize': config.popsize,'timeout': config.timelimit}).optimize(self.evaluation_target).result

            reco = res.xfavorite
            #transforms array into list
            rec = []

            for u in range(reco.size):
                rec.append(reco[u])

        except (RuntimeWarning, RuntimeError, ValueError, ZeroDivisionError, OverflowError, SystemError, AttributeError):
            return False, [0]*scalar_numbers

        return True, rec

    # -------------------------------------------------------------------------------  #
    def best_A_least_squares(self, reco):
    # calls least square fit from cmaes reco :
        try:
            ls_attempt = self.fit(reco)

        except (RuntimeWarning, RuntimeError, ValueError, ZeroDivisionError, OverflowError, SystemError, AttributeError):
            return False, []

        success = ls_attempt.success
        if success:
            reco_ls = ls_attempt.x
            # transforms array into list
            rec = []
            for u in range(reco_ls.size):
                rec.append(reco_ls[u])
            return True, rec

        else:
            return False, []

    # ---------------------------------------------------------------------------- #
    def eval_reward(self, A):
        # given A's, compute the distance to taget function, then calls reward formula:

        derivative_cost = np.zeros(self.n_variables)
        reward = -1

        # this is because derivative cost gives some more numerical error : for validation set, we only use the distance cost
        if self.mode == 'train':
            usederivativecost = 1
        else:
            usederivativecost = 0

        success, result = self.formula_eval(self.variables, A)
        # can fail from two effects : inf/nan from division by zero, or output is a cst, hence a scalar, and not an array

        if success:
            distance_cost = np.sum(np.absolute(result - self.targets))

            for i in range(self.n_variables):

                differential_along_i = np.diff(result - self.targets, axis=i)
                myvar = self.variables[i]
                diff_my_var = np.diff(myvar, axis=i)
                derivative_along_i = np.divide(differential_along_i, diff_my_var)
                derivative_cost[i] = np.sum(np.absolute(derivative_along_i))

            error_on_target = distance_cost + usederivativecost *config.usederivativecost * np.sum(derivative_cost)
            reward = self.reward_formula(error_on_target)

            # add parsimony cost
            if len(A) >= config.parsimony:
                A_cost = config.parsimony_cost * (len(A) - config.parsimony)
                reward -= A_cost
                if reward < -1:
                    reward = -1

        return reward

    # ------------------------------------------------------------------------------- #
    def evaluate(self):
        ''' evaluate the reward of an equation'''

        # Main function:
        # ---------------------------------------------------------------------------- #
        #init stuff
        np.seterr(all = 'ignore')
        allA = []
        failure_reward = -1

        # ---------------------------------------------------------------------------- #
        #rename formulas and return the number of A's
        scalar_numbers = self.rename_formulas()
        #----------------------------------------------------------------------------- #
        # First consider the simple case where there are no generic scalars:
        if scalar_numbers == 0:
            reward = self.eval_reward(allA)
            return reward, scalar_numbers, allA

        # else: cmaes fit : ---------------------------------------------------------- #
        success, allA = self.best_A_cmaes(scalar_numbers)
        if success == False:
            return failure_reward, scalar_numbers, allA

        # ---------------------------------------------------------------------------- #
        #else, compute some actual reward:
        reward_cmaes = self.eval_reward(allA)
        #now we can refine with least squares
        success_ls, allA_ls = self.best_A_least_squares(allA)
        if success_ls == False:
            reward_ls = failure_reward
            reward_ls_round = failure_reward
            allA_ls_round = allA_ls

        # try to round numbers like 3.99999996 to 4 (typical of cmaes)
        else:
            reward_ls = self.eval_reward(allA_ls)

            # and round to the closest significant digit
            allA_ls_round = copy.deepcopy(allA_ls)
            change = False

            c=0
            for a in allA_ls:
                if np.abs(round(a) - a) < 0.1:
                    allA_ls_round[c] = round(a)
                    change = True

                c+=1

            if change:
                reward_ls_round = self.eval_reward(allA_ls_round)
            else:
                reward_ls_round = -2

        #now compare the three and chose the best
        allrewards=[reward_cmaes, reward_ls, reward_ls_round]
        m = max(allrewards)
        best = [i for i, j in enumerate(allrewards) if j == m][0]

        if best == 0:
            return reward_cmaes, scalar_numbers, allA
        elif best == 1 and reward_ls > reward_ls_round:
            return reward_ls, scalar_numbers, allA_ls
        else:
            return reward_ls_round, scalar_numbers, allA_ls_round
