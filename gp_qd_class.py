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
from multiprocessing import Pool


# ============================  QD version ====================================#

# ---------------------------------------------------------------------------- #
class GP_QD():

    # ---------------------------------------------------------------------------- #
    def __init__(self, target_number, delete_ar1_ratio, p_mutate, p_cross, poolsize, voc, tolerance,
                  extend_ratio, maxa, bina, maxl, binl, maxf, binf, maxp, binp, maxtrig, bintrig, maxexp, binexp, addrandom, qdpool, pool = None):

        self.target_number = target_number
        self.usesimplif = config.use_simplif
        self.p_mutate = p_mutate
        self.delete_ar1_ratio = delete_ar1_ratio
        self.p_cross= p_cross
        self.poolsize = poolsize
        self.pool = pool
        self.QD_pool = qdpool
        self.pool_to_eval = []
        self.tolerance = tolerance
        self.maximal_size = voc.maximal_size
        self.extend = extend_ratio
        self.addrandom = addrandom
        self.voc = voc
        self.maxa = maxa
        self.bina = bina
        self.maxl = maxl
        self.binl = binl
        self.maxf = maxf
        self.binf = binf
        self.maxp = maxp
        self.binp = binp

        self.maxtrig = maxtrig
        self.bintrig = bintrig

        self.maxexp = maxexp
        self.binexp = binexp
        self.smallstates =[]

    def par_crea(self, task):
        np.random.seed(task)
        newgame = game_env.randomeqs(self.voc)
        return newgame.state

    def para_complet(self, task):
        st = time.time()
        #np.random.seed(task)
        #index = np.random.randint(0, len(self.smallstates))
        #print(index)

        #print('before', self.smallstates[index].formulas)
        #newstate = game_env.complete_eq_with_random(self.voc, self.smallstates[task])
        #print(time.time() -st)
        #print('after', newstate.formulas)

        #return newstate
        return 1
    # ---------------------------------------------------------------------------- #
    #creates or extend self.pool
    def extend_pool(self):
        # init : create random eqs
        if self.pool == None and self.QD_pool is None:
            multi = True
            if multi:
                self.pool = []
                tasks = range(0, self.poolsize)
                # print(tasks)
                mp_pool = mp.Pool(config.cpus)
                asyncResult = mp_pool.map_async(self.par_crea, tasks)
                results = asyncResult.get()
                mp_pool.close()
                mp_pool.join()
                print('yo', len(results))
                for state in results:
                    if self.voc.infinite_number not in state.reversepolish:
                        self.pool.append(state)
                print('yu', len(self.pool))

            else:
                self.pool = []

                while len(self.pool) < self.poolsize:
                    newgame = game_env.randomeqs(self.voc)
                    newrdeq = newgame.state

                    if self.voc.infinite_number not in newrdeq.reversepolish:
                        self.pool.append(newrdeq)

            return self.pool


        else:
            gp_motor = generate_offsprings(self.delete_ar1_ratio, self.p_mutate, self.p_cross, self.maximal_size, self.voc, self.maximal_size)

            all_states = []
            small_states=[]
            for bin_id in self.QD_pool:
                all_states.append(self.QD_pool[str(bin_id)][1])
                if eval(bin_id)[1] < self.maximal_size -10:
                    small_states.append(self.QD_pool[str(bin_id)][1])

            self.smallstates = small_states

            # +add new rd eqs for diversity. We add half the qd_pool size of random eqs
            if self.addrandom:
                toadd = int(len(self.QD_pool)/2)
                c=0
                ntries = 0
                trypara = False
                if trypara:
                    st = time.time()
                    tasks = range(0, toadd)
                    print('ici', toadd)
                    st= time.time()
                    mp_pool = mp.Pool(config.cpus)
                    print('y', st - time.time())
                    asyncResult = mp_pool.map_async(self.para_complet, tasks)
                    print('yy', st - time.time())

                    results = asyncResult.get(timeout=1)
                    print('yyy' , st - time.time())

                    mp_pool.close()
                    mp_pool.join()
                    for res in results:
                        if self.voc.infinite_number not in res.reversepolish:
                            all_states.append(res)

                else:
                    st = time.time()

                    while c < toadd and ntries < 10000:
                        index = np.random.randint(0, len(small_states))
                        state = small_states[index]
                        newstate = game_env.complete_eq_with_random(self.voc, state)
                        if self.voc.infinite_number not in newstate.reversepolish:
                            all_states.append(newstate)
                            c+=1
                        ntries+=1
                print('toi complet', time.time() -st)


            #print('trying', ntries, 'vs', toadd)
            ts = time.time()
            print('sizetocross', len(self.QD_pool))
            #then mutate and crossover
            newpool = []
            count=0
            while len(newpool) < int(self.extend*len(self.QD_pool)) and count < 400000:
                index = np.random.randint(0, len(all_states))
                state = all_states[index]
                u = random.random()

                if u <= self.p_mutate:
                    count += 1

                    mutatedstate = gp_motor.mutate(state)
                    #if str(mutatedstate.reversepolish) not in alleqs:
                    newpool.append(mutatedstate)

                elif u <= self.p_cross:
                    count += 2

                    index = np.random.randint(0, len(all_states))
                    otherstate = all_states[index]  # this might crossover with itself : why not!
                    state1, state2 = gp_motor.crossover(state, otherstate)

                    #if str(state1.reversepolish) not in alleqs:
                    newpool.append(state1)
                    #if str(state2.reversepolish) not in alleqs:
                    newpool.append(state2)

                else:  # mutate AND cross
                    count += 2

                    index = np.random.randint(0, len(all_states))
                    to_mutate = copy.deepcopy(all_states[index])
                    prestate1 = gp_motor.mutate(state)
                    prestate2 = gp_motor.mutate(to_mutate)
                    state1, state2 = gp_motor.crossover(prestate1, prestate2)

                    #if str(state1.reversepolish) not in alleqs:
                    newpool.append(state1)
                    #if str(state2.reversepolish) not in alleqs:
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
        print('je starte', len(results))
        ct=0
        cif=0
        cnotif=0
        # rescale and print which bin
        for oneresult in results:
            reward, state, allA, Anumber, L, function_number, powernumber, trignumber, explognumber = oneresult

            #if state.reversepolish[-1] == 1:
            #    L = L - 1

            #needs to be computed if mode 'noA'
            #if self.voc.modescalar == 'noA':
            #    Anumber = 0
            #    for char in state.reversepolish:
            #        if char in self.voc.pure_numbers:
            #            Anumber+=1

            if Anumber >= self.maxa:
                bin_a = self.bina
            else:
                bins_for_a = np.linspace(0, self.maxa, num=self.bina+1)
                for i in range(len(bins_for_a) -1):
                    if Anumber >= bins_for_a[i] and Anumber < bins_for_a[i + 1]:
                        bin_a = i

            if L >= self.maxl:
                bin_l = self.binl
            else:
                bins_for_l = np.linspace(0, self.maxl, num=self.binl+1)

                for i in range(len(bins_for_l) - 1):
                    if L >= bins_for_l[i] and L < bins_for_l[i + 1]:
                        bin_l = i

            if function_number >= self.maxf:
                bin_f = self.binf
            else:
                bins_for_f = np.linspace(0, self.maxf, num = self.binf+1)
                for i in range(len(bins_for_f) - 1):
                    if function_number >= bins_for_f[i] and function_number < bins_for_f[i + 1]:
                        bin_f = i

            if powernumber >= self.maxp:
                bin_p = self.binp
            else:
                bins_for_p = np.linspace(0, self.maxp, num=self.binp + 1)
                for i in range(len(bins_for_p) - 1):
                    if powernumber >= bins_for_p[i] and powernumber < bins_for_p[i + 1]:
                        bin_p = i

            if trignumber >= self.maxtrig:
                bin_trig = self.bintrig
            else:
                bins_for_trig = np.linspace(0, self.maxtrig, num=self.bintrig + 1)
                for i in range(len(bins_for_trig) - 1):
                    if trignumber >= bins_for_trig[i] and trignumber < bins_for_trig[i + 1]:
                        bin_trig = i

            if explognumber >= self.maxexp:
                bin_exp = self.binexp
            else:
                bins_for_exp = np.linspace(0, self.maxexp, num=self.binexp + 1)
                for i in range(len(bins_for_exp) - 1):
                    if explognumber >= bins_for_exp[i] and explognumber < bins_for_exp[i + 1]:
                        bin_exp = i
        #    print('entry', reward, state, allA, Anumber, L, function_number, powernumber, trignumber, explognumber)

         #   print(state.reversepolish, state.formulas)
          #  print('bins are', [bin_a, bin_l, bin_exp, bin_trig, bin_p, bin_f])
            ct+=1
            if str([bin_a, bin_l, bin_exp, bin_trig, bin_p, bin_f]) not in results_by_bin:
                cif+=1
                results_by_bin.update({str([bin_a, bin_l, bin_exp, bin_trig, bin_p, bin_f]): [reward, state, allA]})
                #print(results_by_bin)
            else:
                cnotif+=1
                #print('happening', [bin_a, bin_l, bin_exp, bin_trig, bin_p, bin_f])
                #time.sleep(.2)
                #print(state.formulas)
                prev_reward = results_by_bin[str([bin_a, bin_l, bin_exp, bin_trig, bin_p, bin_f])][0]
                #print('prev', results_by_bin[str([bin_a, bin_l, bin_exp, bin_trig, bin_p, bin_f])])
                #print('prev', results_by_bin[str([bin_a, bin_l, bin_exp, bin_trig, bin_p, bin_f])][1].formulas)
                if reward > prev_reward:
                    results_by_bin.update({str([bin_a, bin_l, bin_exp, bin_trig, bin_p, bin_f]): [reward, state, allA]})
        #print(results_by_bin)
        #print('me', len(results_by_bin))
        print(ct, cif, cnotif)
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
                prev_reward = self.QD_pool[binid][0]
                reward = newresults_by_bin[binid][0]
                if reward > prev_reward:
                    self.QD_pool.update({binid: newresults_by_bin[binid]})
                    replacement += 1
        print('news', newbin, replacement)
        return newbin, replacement

# ========================   end class gp_qd ====================== #

class printresults():

    def __init__(self, target, voc):
        self.target = target
        self.voc = voc

    # ---------------------------------------------- |
    # to print understandable results
    def finalrename(self, bestform, A):

        formula = bestform

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

            rename = rename.replace('np.', '')
            rename = rename.replace('x0', 'x')
            rename = rename.replace('x1', 'y')
            rename = rename.replace('x2', 'z')

            #handle le plus one
            if A_count < len(As):
                print('this is obsolete')
                rename += '+ A[' + str(A_count) + ']'

                for i in range(A_count + 1):
                    to_replace = 'A[' + str(i) + ']'
                    replace_by = '(' + str(As[i]) + ')'
                    rename = rename.replace(to_replace, replace_by)
            else:

                for i in range(A_count):
                    to_replace = 'A[' + str(i) + ']'
                    replace_by = '(' + str(As[i]) + ')'
                    rename = rename.replace(to_replace, replace_by)
        else:
            formula = formula.replace('np.', '')
            formula = formula.replace('x0', 'x')
            formula = formula.replace('x1', 'y')
            formula = formula.replace('x2', 'z')
            rename = formula

        return rename


    def saveresults(self, newbin, replacements, i, QD_pool, maxa, tolerance, target_number, alleqs, prefix):


        # rank by number of free parameters
        bests = []

        for a in range(maxa):
            eqs = []
            for bin_id in QD_pool:
                anumber = int(bin_id.split(',')[0].replace('[', ''))
                if anumber == a:
                    eqs.append(QD_pool[str(bin_id)])

            if len(eqs) > 0:
                sort = sorted(eqs, key=itemgetter(0), reverse=True)
                thebest = sort[0]
                thebestformula = thebest[1].formulas
                thebest_as = thebest[2]
                bests.append([int(1000 * thebest[0]) / 1000, self.finalrename(thebestformula, thebest_as)])

        # best of all
        all_states = []
        for bin_id in QD_pool:
            all_states.append(QD_pool[str(bin_id)])

        rank = sorted(all_states, key=itemgetter(0), reverse=True)
        best_state = rank[0][1]
        with_a_best = rank[0][2]
        best_formula = best_state.formulas
        bestreward = rank[0][0]

        if np.isnan(bestreward) or np.isinf(bestreward):
            bestreward=-1

        evaluate = Evaluatefit(best_formula, self.voc, self.target, tolerance, 'test')
        evaluate.rename_formulas()

        validation_reward = evaluate.eval_reward(with_a_best)
        if np.isnan(validation_reward) or np.isinf(validation_reward):
            validation_reward = -1

        useful_form = self.finalrename(best_formula, with_a_best)

        if bestreward > 0.999:
            print(best_formula, with_a_best)
            print(evaluate.formulas)
            print(validation_reward)
            print('again')
            validation_reward = evaluate.eval_reward(with_a_best)
            if np.isnan(validation_reward) or np.isinf(validation_reward):
                validation_reward = -1
            print('val2', validation_reward)
        if bestreward == 1.0:
            validation_reward = 1.0
        #other statistics
        avgreward = 0
        for x in rank:
            reward = x[0]
            if np.isnan(reward) or np.isinf(reward):
                reward=-1
            avgreward += reward

        avgreward = avgreward / len(rank)
        #avg validation reward:
        avg_validation_reward = 0
        for x in rank:
            state = x[1]
            with_a = x[2]

            formula = state.formulas
            evaluate = Evaluatefit(formula, self.voc, self.target, tolerance, 'test')
            evaluate.rename_formulas()
            avg_validation_reward += evaluate.eval_reward(with_a)
        avg_validation_reward /= len(rank)

        timespent = time.time() - eval(prefix)/10000000
        if config.uselocal:
            filepath = './' + prefix + 'results_target_' + str(target_number) + '.txt'
        else:
            filepath = '/home/user/results/'+ prefix+ 'results_target_' + str(target_number) + '.txt'
        with open(filepath, 'a') as myfile:

            myfile.write('iteration ' + str(i) + ': we have seen ' + str(alleqs) + ' different eqs')
            myfile.write("\n")
            myfile.write(
                'QD pool size: ' + str(len(QD_pool)) + ', newbins: ' + str(newbin) + ' replacements: ' + str(
                    replacements))
            myfile.write("\n")
            myfile.write("\n")

            myfile.write('new avg training reward: ' + str(int(1000 * avgreward) / 1000))
            myfile.write(' new avg validation reward: ' + str(int(1000 * avg_validation_reward) / 1000))
            myfile.write("\n")

            myfile.write('best reward: ' + str(int(10000 * bestreward) / 10000) + ' with validation reward: ' + str(
                validation_reward))
            myfile.write("\n")
            myfile.write("\n")

            myfile.write('best eq: ' + str(useful_form) + ' ' + str(best_formula) + ' ' + str(with_a_best))
            myfile.write("\n")
            myfile.write("\n")

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

        return validation_reward

