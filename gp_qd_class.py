#  ======================== CMA-Based Symbolic Regressor ========================== #
# Project:          Symbolic regression for physics
# Name:             AST.py
# Authors:          Jean-Philippe Bruneton
# Date:             2020
# License:          BSD 3-Clause License
# ============================================================================ #


# ================================= PREAMBLE ================================= #
# Packages
import numpy as np
import copy
import random
import config
from operator import itemgetter
from generate_offsprings import generate_offsprings
from Evaluate_fit import Evaluatefit
import time
import game_env
import multiprocessing as mp
from game_env import Game


# ============================  QD version ====================================#

# ---------------------------------------------------------------------------- #
class GP_QD():

    def __init__(self, delete_ar1_ratio, delete_ar2_ratio, p_mutate, p_cross, poolsize, voc,
                 extend_ratio, bin_length, bin_nscalar, bin_functions, bin_depth, bin_power, bin_target,
                 bin_derivative, bin_var, bin_norm, bin_cross, bin_dot_product,
                 addrandom, calculus_mode, maximal_size, qdpool, pool = None):

        self.calculus_mode = calculus_mode
        self.usesimplif = config.use_simplif
        self.p_mutate = p_mutate
        self.delete_ar1_ratio = delete_ar1_ratio
        self.delete_ar2_ratio = delete_ar2_ratio

        self.p_cross= p_cross
        self.poolsize = poolsize
        self.pool = pool
        self.QD_pool = qdpool
        self.pool_to_eval = []
        self.maximal_size = maximal_size
        self.extend = extend_ratio
        self.addrandom = addrandom
        self.voc = voc
        self.bin_nscalar = bin_nscalar
        self.binl = bin_length
        self.binf = bin_functions
        self.binp = bin_power
        self.bin_depth = bin_depth
        self.bin_target = bin_target
        self.bin_derivatives = bin_derivative
        self.bin_norm = bin_norm
        self.bin_var = bin_var
        self.bin_cross = bin_cross
        self.bin_dot = bin_dot_product
        self.smallstates = []

    # ----------------------
    def parallel_creation(self, task):
        np.random.seed(task)
        newgame = game_env.randomeqs(self.voc)
        return newgame.state

    # ---------------------------------------------------------------------------- #
    # creates or extend self.pool
    def extend_pool(self):

        if self.pool == None and self.QD_pool is None:
            self.pool = []
            tasks = range(0, self.poolsize)
            mp_pool = mp.Pool(config.cpus)
            asyncResult = mp_pool.map_async(self.parallel_creation, tasks)
            results = asyncResult.get()
            mp_pool.close()
            mp_pool.join()
            for state in results:
                if self.voc.infinite_number[0] not in state.reversepolish:
                    self.pool.append(state)
            del mp_pool
            return self.pool

        else:
            gp_motor = generate_offsprings(self.delete_ar1_ratio, self.delete_ar1_ratio, self.p_mutate,
                                           self.p_cross, self.maximal_size, self.voc, self.calculus_mode)
            all_states = []
            small_states = []
            for bin_id in self.QD_pool:
                all_states.append(self.QD_pool[str(bin_id)][1])
                if eval(bin_id)[1] < self.maximal_size - 10:
                    # we call small states equations that have 10 character or more less than the maximal size
                    small_states.append(self.QD_pool[str(bin_id)][1])

            self.smallstates = small_states

            # +add new rd eqs for diversity. We add half the qd_pool size of random eqs
            if self.addrandom or self.maximal_size < 20:
                toadd = int(len(self.QD_pool)/2)
                c = 0
                n_tries = 0

                st = time.time()

                while c < toadd and n_tries < 2000:
                    newgame = game_env.randomeqs(self.voc)
                    if self.voc.infinite_number[0] not in newgame.state.reversepolish:
                        all_states.append(newgame.state)
                        c += 1
                    n_tries += 1
                print('completion random duration', time.time() -st)

            ts = time.time()

            print('how many states to mutate/cross:', len(self.QD_pool))

            # then mutate and crossover
            newpool = []
            count = 0
            treshold = int(self.extend * len(self.QD_pool))

            while len(newpool) < treshold and count < 400000:
                index = np.random.randint(0, len(all_states))
                state = all_states[index]
                u = random.random()

                if u <= self.delete_ar2_ratio and self.calculus_mode == 'vectorial':
                    count += 1
                    s, newstate = gp_motor.vectorial_delete_one_subtree(state)
                    newpool.append(newstate)

                else:
                    if u <= self.p_mutate:
                        count += 1
                        if self.calculus_mode == 'scalar':
                            s, mutatedstate = gp_motor.mutate(state)
                        else:
                            s, mutatedstate = gp_motor.vectorial_mutation(state)

                        # if str(mutatedstate.reversepolish) not in alleqs:
                        newpool.append(mutatedstate)

                    elif u <= self.p_cross:
                        count += 2

                        index = np.random.randint(0, len(all_states))
                        otherstate = all_states[index]  # this might crossover with itself : why not!
                        if self.calculus_mode == 'scalar':
                            success, state1, state2 = gp_motor.crossover(state, otherstate)
                        else:
                            success, state1, state2 = gp_motor.vectorial_crossover(state, otherstate)

                        if success:
                            # if str(state1.reversepolish) not in alleqs:
                            newpool.append(state1)
                            # if str(state2.reversepolish) not in alleqs:
                            newpool.append(state2)

                    else:  # mutate AND cross
                        count += 2

                        index = np.random.randint(0, len(all_states))
                        to_mutate = copy.deepcopy(all_states[index])
                        if self.calculus_mode == 'scalar':
                            s, prestate1 = gp_motor.mutate(state)
                            s, prestate2 = gp_motor.mutate(to_mutate)
                            suc, state1, state2 = gp_motor.crossover(prestate1, prestate2)
                        else:
                            s, prestate1 = gp_motor.vectorial_mutation(state)
                            s, prestate2 = gp_motor.vectorial_mutation(to_mutate)
                            suc, state1, state2 = gp_motor.vectorial_crossover(prestate1, prestate2)
                        if suc:
                            # if str(state1.reversepolish) not in alleqs:
                            newpool.append(state1)
                            # if str(state2.reversepolish) not in alleqs:
                            newpool.append(state2)

            print('avgtime', (time.time()-ts))
            #update self.pool
            self.pool = newpool
            print('yo', len(newpool))

            return self.pool

    # ---------------------------------------------------------------------------- #
    # bin the results
    def bin_pool(self, results):

        results_by_bin = {}

        for oneresult in results:
            rms, state, allA, Anumber = oneresult
            game = Game(self.voc, state)
            if self.calculus_mode == 'scalar':
                L, function_number, mytargetnumber, firstder_number, depth, varnumber = game.get_features()
            else:
                L, function_number, mytargetnumber, firstder_number, depth, varnumber, \
                dotnumber, normnumber, crossnumber = game.get_features()

            bins = np.linspace(0, self.bin_nscalar, num=self.bin_nscalar+1)
            bin_scalar = np.digitize(Anumber, bins)

            bins = np.linspace(0, self.binl, num=self.binl + 1)
            bin_l = np.digitize(L, bins)

            bins = np.linspace(0, self.binf, num=self.binf + 1)
            bin_f = np.digitize(function_number, bins)

            bins = np.linspace(0, self.bin_depth, num=self.bin_depth + 1)
            bin_depth = np.digitize(depth, bins)

            bins = np.linspace(0, self.bin_target, num=self.bin_target + 1)
            bin_targ = np.digitize(mytargetnumber, bins)

            bins = np.linspace(0, self.bin_derivatives, num=self.bin_derivatives + 1)
            bin_der = np.digitize(firstder_number, bins)

            bins = np.linspace(0, self.bin_var, num=self.bin_var + 1)
            bin_var = np.digitize(varnumber, bins)


            if self.calculus_mode == 'vectorial':

                bins = np.linspace(0, self.bin_dot, num=self.bin_dot + 1)
                bin_dot = np.digitize(dotnumber, bins)

                bins = np.linspace(0, self.bin_norm, num=self.bin_norm + 1)
                bin_norm = np.digitize(normnumber, bins)

                bins = np.linspace(0, self.bin_cross, num=self.bin_cross + 1)
                bin_cross = np.digitize(crossnumber, bins)


            if self.calculus_mode =='scalar':
                if str([bin_scalar, bin_l, bin_f, bin_depth, bin_targ, bin_der, bin_var]) not in results_by_bin:
                    if rms <config.minrms:
                        results_by_bin.update({str([bin_scalar, bin_l, bin_f, bin_depth, bin_targ, bin_der, bin_var]): [rms, state, allA]})
                else:
                    prev_rms = results_by_bin[str([bin_scalar, bin_l, bin_f, bin_depth, bin_targ, bin_der, bin_var])][0]
                    if rms < prev_rms:
                        results_by_bin.update({str([bin_scalar, bin_l, bin_f, bin_depth, bin_targ, bin_der, bin_var]): [rms, state, allA]})
            else:
                if str([bin_scalar, bin_l, bin_f, bin_depth, bin_targ, bin_der, bin_var, bin_dot, bin_norm, bin_cross]) not in results_by_bin:
                    if rms <config.minrms:
                        results_by_bin.update({str([bin_scalar, bin_l, bin_f, bin_depth, bin_targ, bin_der, bin_var, bin_dot, bin_norm, bin_cross]): [rms, state, allA]})
                else:
                    prev_rms = results_by_bin[str([bin_scalar, bin_l, bin_f, bin_depth, bin_targ, bin_der, bin_var, bin_dot, bin_norm, bin_cross])][0]
                    if rms < prev_rms:
                        results_by_bin.update({str([bin_scalar, bin_l, bin_f, bin_depth, bin_targ, bin_der, bin_var, bin_dot, bin_norm, bin_cross]): [rms, state, allA]})

        return results_by_bin

    # ---------------------------------------------------------------------------- #
    #updtae qd_pool according to new results
    def update_qd_pool(self, newresults_by_bin):
        newbin = 0
        replacement = 0

        for binid in newresults_by_bin:
            if binid not in self.QD_pool:
                self.QD_pool.update({binid: newresults_by_bin[binid]})
                newbin += 1
            else:
                prev_rms = self.QD_pool[binid][0]
                rms = newresults_by_bin[binid][0]
                if rms < prev_rms:
                    self.QD_pool.update({binid: newresults_by_bin[binid]})
                    replacement += 1
        print('new bins and replacements', newbin, replacement)
        return newbin, replacement

# ========================   end class gp_qd ====================== #

class printresults():

    def __init__(self, target, voc, calculusmode, lookfor, name):
        self.target = target
        self.calculusmode = calculusmode
        self.name = name

        if self.calculusmode == 'scalar':
            self.n_variables = len(self.target[0][5]) - 1
        else:
            self.n_variables = 1

        self.ranges = self.target[0][5]
        self.voc = voc
        self.lookfor = lookfor

    # ---------------------------------------------- |
    # to print understandable results
    def finalrename(self, bestform, A):

        formula = bestform
        string_to_replace = 'B'
        replace_by = 'np.array([A,A,A])'
        formula = formula.replace(string_to_replace, replace_by)

        # round for lisibility
        As = [int(1000000*x)/1000000 for x in A]
        if As != []:
            rename = ''
            A_count = 0
            for char in formula:
                if char == 'A':
                    rename += 'A[' + str(A_count) + ']'
                    A_count += 1
                else:
                    rename += char

            if A_count < len(As): #the case where we had to add one scalar to the cma
                rename += '+ A[' + str(A_count) + ']'
                A_count+=1

            for i in range(A_count):
                to_replace = 'A[' + str(i) + ']'
                replace_by = str(As[i])
                rename = rename.replace(to_replace, replace_by)

        else:
            rename = formula

        if self.calculusmode == 'scalar':
            # re-scale the scaled_f and scale_x:
            rename = rename.replace('d_x0_f0', '(' + str(self.ranges[0]) + ')/(' + str(self.ranges[self.n_variables])+')' +'*(df(x))')
            rename = rename.replace('f0', 'f(x)')
            rename = rename.replace('x0', 'x/' + str(self.ranges[0]))
            if self.n_variables > 1:
                rename = rename.replace('x1', 'y/'+str(self.ranges[1]))
            if self.n_variables > 2:
                rename = rename.replace('x2', 'z/'+str(self.ranges[2]))

        else:
            prefactor = 'np.array(['+ str(self.ranges[0]) + '/' + str(self.ranges[1]) \
                        + ',' + str(self.ranges[0]) + '/' + str(self.ranges[2]) \
                        + ',' + str(self.ranges[0]) + '/' + str(self.ranges[3]) + '])'
            p1 = 'np.array([' + str(self.ranges[1]) \
                        + ',' + str(self.ranges[2]) \
                        + ',' + str(self.ranges[3]) + '])'
            p2 = 'np.array([' + str(self.ranges[1]) + '/' + str(self.ranges[0]) \
                        + ',' + str(self.ranges[2]) + '/' + str(self.ranges[0]) \
                        + ',' + str(self.ranges[3]) + '/' + str(self.ranges[0]) + '])'
            p3 = 'np.array([' + str(self.ranges[1]) + '/' + str(self.ranges[0])+'**2' \
                        + ',' + str(self.ranges[2]) + '/' + str(self.ranges[0])+'**2' \
                        + ',' + str(self.ranges[3]) + '/' + str(self.ranges[0]) +'**2' + '])'
            rename = rename.replace('d_x0_F0', 'np.multiply('+ prefactor + ', df(x))')
            rename = rename.replace('f0', 'f(x)')
            rename = rename.replace('x0', 't/'+str(self.ranges[0]))
            if self.lookfor == 'find_function':
                rename = 'np.multiply(' + p1 + ',' + rename + ')'
            elif self.lookfor == 'find_1st_order_diff_eq': #its only 1D diff eq
                rename = 'np.multiply(' + p2 + ',' + rename + ')'
            elif self.lookfor == 'find_2nd_order_diff_eq': #its only 1D diff eq
                rename = 'np.multiply(' + p3 + ',' + rename + ')'

        #rescale global scaled_f
        if self.calculusmode == 'scalar':
            if self.lookfor == 'find_function':
                rename = str(self.ranges[self.n_variables]) +'*(' + rename + ')'
            elif self.lookfor == 'find_1st_order_diff_eq': #its only 1D diff eq
                rename = str(self.ranges[self.n_variables]) +'*(' + rename + ')/'+str(self.ranges[0])
            elif self.lookfor == 'find_2nd_order_diff_eq': #its only 1D diff eq
                rename = '(' + str(self.ranges[self.n_variables]) + ')/(' + str(self.ranges[0])+'**2)' +'*(' + rename + ')'

        # .reshape SIZE not useful either :
        rename = rename.replace('axis=1).reshape(SIZE, 1)', 'axis=0)')
        #rename = rename.replace('np.', '')

        return rename

    def saveresults(self, newbin, replacements, i, QD_pool, maxa, alleqs, sttime, u, look_for):

        # rank by number of free parameters
        bests = []
        best_simplified_formulas = []


        for a in range(maxa):
            eqs = []
            for bin_id in QD_pool:
                anumber = int(bin_id.split(',')[0].replace('[', ''))
                if anumber == a:
                    eqs.append(QD_pool[str(bin_id)])

            if len(eqs) > 0:
                sort = sorted(eqs, key=itemgetter(0), reverse=False)
                thebest = sort[0]
                thebestformula = thebest[1].formulas
                thebest_as = thebest[2]
                #simple = game_env.simplif_eq(self.voc, thebest[1])
                #best_simplified_formulas.append(simple.formulas)
                bests.append([thebest[0], self.finalrename(thebestformula, thebest_as)])

        # best of all
        all_states = []
        for bin_id in QD_pool:
            all_states.append(QD_pool[str(bin_id)])

        rank = sorted(all_states, key=itemgetter(0), reverse=False)
        best_state = rank[0][1]
        with_a_best = rank[0][2]
        best_formula = best_state.formulas
        bestreward = rank[0][0]

        if np.isnan(bestreward) or np.isinf(bestreward):
            bestreward = 100000000
        evaluate = Evaluatefit(best_formula, self.voc, self.target, 'train', u, look_for)
        evaluate.rename_formulas()

        if self.calculusmode == 'scalar':
            validation_reward = evaluate.eval_reward_nrmse(with_a_best)
        else:
            validation_reward = evaluate.eval_reward_nrmse_vectorial(with_a_best)

        if validation_reward > 100000000:
            validation_reward = 100000000

        if np.isnan(validation_reward) or np.isinf(validation_reward):
            validation_reward = 100000000

        useful_form = self.finalrename(best_formula, with_a_best)

        if bestreward < config.termination_nmrse:
            print(best_formula, with_a_best)
            print(evaluate.formulas)
            print(validation_reward)

        if bestreward < config.termination_nmrse:
            validation_reward = 0.
        #other statistics
        avgreward = 0
        for x in rank:
            reward = x[0]
            if np.isnan(reward) or np.isinf(reward):
                reward=100000000
            avgreward += reward

        avgreward = avgreward / len(rank)
        #avg validation reward:
        avg_validation_reward = 0
        for x in rank:
            state = x[1]
            with_a = x[2]

            formula = state.formulas

            evaluate = Evaluatefit(formula, self.voc, self.target, 'train',u, look_for)
            evaluate.rename_formulas()
            if self.calculusmode == 'scalar':
                avg_validation_reward += evaluate.eval_reward_nrmse(with_a)
            else:
                avg_validation_reward += evaluate.eval_reward_nrmse_vectorial(with_a)

        avg_validation_reward /= len(rank)

        timespent = time.time() - sttime
        if config.uselocal:
            filepath = './results/results_target_' + self.name + '.txt'
        else:
            filepath = '/home/user/results/results_target_' + self.name + '.txt'
        with open(filepath, 'a') as myfile:

            myfile.write('iteration ' + str(i) + ': we have seen ' + str(len(alleqs)) + ' different eqs')
            myfile.write("\n")
            myfile.write(
                'QD pool size: ' + str(len(QD_pool)) + ', newbins: ' + str(newbin) + ' replacements: ' + str(
                    replacements))
            myfile.write("\n")
            myfile.write("\n")

            myfile.write('new avg training reward: ' + str(int(1000 * avgreward) / 1000))
#            myfile.write(' new avg validation reward: ' + str(int(1000 * avg_validation_reward) / 1000))
            myfile.write("\n")

            myfile.write('best reward: ' + str(bestreward) + ' with validation reward: ' + str(validation_reward))
            myfile.write("\n")
            myfile.write("\n")

            myfile.write('best eq: ' + str(useful_form) + ' ' + str(best_formula) + ' ' + str(with_a_best))
            myfile.write("\n")
            myfile.write("\n")
            if False:
                x = Symbol('x')
                try:
                    test = series(useful_form, x)
                    myfile.write('best eq DL en zero:' +str(test))
                    myfile.write("\n")
                    myfile.write("\n")
                except Exception as e:
                    pass
                useful_form = useful_form.replace('exp', 'epon')
                useful_form = useful_form.replace('x', '(1/x)')
                useful_form = useful_form.replace('epon', 'exp')
                try:
                    test = series(useful_form, x)
                    test = test.replace('epon', 'exp')

                    myfile.write('best eq DL en inf:' +str(test))
                    myfile.write("\n")
                    myfile.write("\n")
                except Exception as e:
                    pass
            myfile.write('and bests eqs by free parameter number are:')
            myfile.write("\n")
            myfile.write(str(bests))
            myfile.write("\n")
            myfile.write("\n")
            myfile.write('time spent (in secs):' + str(timespent))
            myfile.write("\n")
            myfile.write("\n")
            myfile.write("---------------=============================----------------")
            myfile.write("\n")

            myfile.close()

        return validation_reward, best_simplified_formulas

# #restes                 if config.specialgal:
#                     if config.loglog:
#                         rename = rename.replace('x0', 'x - np.log10(A[' + str(A_count) + '])')
#                     else:
# #                         rename = rename.replace('x0', 'x/A[' + str(A_count) + ']')
# #                     A_count+=1
# if config.specialgal:
#     if config.loglog:
#         rename = rename + '+np.log10(A[' + str(A_count - 1) + '])'
#     else:
#         rename = rename + '*A[' + str(A_count - 1) + ']'

