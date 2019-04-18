from game_env import Game
import numpy as np
import copy
import random
import config
from operator import itemgetter
from Targets import Target, Voc
from generate_offsprings import generate_offsprings
from Evaluate_fit import Evaluatefit
import game_env
import os
import multiprocessing as mp
import time
import csv


class GP():
    # ---------------------------------------------------------------------------- #
    def __init__(self, delete_ar1_ratio, p_mutate, p_cross, maximal_size, poolsize,  target, tolerance, extend_ratio, voc, addrandom, pool = None):

        self.usesimplif = config.use_simplif
        self.p_mutate = p_mutate
        self.delete_ar1_ratio = delete_ar1_ratio
        self.p_cross= p_cross
        self.poolsize = poolsize
        self.pool = pool
        self.target = target
        self.tolerance = tolerance
        self.maximal_size = maximal_size
        self.extend = extend_ratio
        self.voc = voc
        self.addrandom = addrandom

    def par_crea(self, task):
        np.random.seed(task)
        newgame = game_env.randomeqs(self.voc)
        return newgame.state
    # ---------------------------------------------------------------------------- #
    # init a new pool of rd eqs if doesnt exists, or extend it by mutations/crossovers

    def extend_pool(self):

        if self.pool == None:
            self.pool = []
            tasks = range(0, self.poolsize)

            mp_pool = mp.Pool(config.cpus)
            asyncResult = mp_pool.map_async(self.par_crea, tasks)
            results = asyncResult.get()
            mp_pool.close()
            mp_pool.join()

            for state in results:
                if self.voc.infinite_number not in state.reversepolish:
                    self.pool.append(state)

            return self.pool

        #extend the pool by a factor 2, say
        else:
            gp_motor = generate_offsprings(self.delete_ar1_ratio, self.p_mutate, self.p_cross, self.maximal_size, self.voc, self.maximal_size)

            all_states = []

            small_states=[]
            for state in self.pool:
                all_states.append(state)
                if len(state.reversepolish) < self.maximal_size - 10:
                    small_states.append(state)


            # +add new rd eqs for diversity. We add half the qd_pool size of random eqs
            if self.addrandom:
                toadd = int(len(all_states)/2)
                c=0
                ntries = 0
                if len(small_states) > 10:
                    while c < toadd and ntries < 10000:
                        index = np.random.randint(0, len(small_states))
                        state = small_states[index]
                        newstate = game_env.complete_eq_with_random(self.voc, state)
                        if self.voc.infinite_number not in newstate.reversepolish:
                            all_states.append(newstate)
                            c+=1
                        ntries+=1
                else:
                    while c < toadd :
                        newstate = game_env.randomeqs(self.voc).state
                        if self.voc.infinite_number not in newstate.reversepolish:
                            all_states.append(newstate)
                            c+=1



            print('sizetocross', len(all_states))

            #then mutate and crossover
            newpool = []
            count=0
            while len(newpool) < int(self.extend*self.poolsize) and count < 40000:
                index = np.random.randint(0, len(all_states))
                state = all_states[index]
                u = random.random()

                if u <= self.p_mutate:
                    count += 1
                    mutatedstate = gp_motor.mutate(state)
                    newpool.append(mutatedstate)

                elif u <= self.p_cross:
                    count += 2
                    index = np.random.randint(0, len(all_states))
                    otherstate = all_states[index]  # this might crossover with itself : why not!
                    state1, state2 = gp_motor.crossover(state, otherstate)
                    newpool.append(state1)
                    newpool.append(state2)

                else:  # mutate AND cross
                    count += 2

                    index = np.random.randint(0, len(all_states))
                    to_mutate = copy.deepcopy(all_states[index])
                    prestate1 = gp_motor.mutate(state)
                    prestate2 = gp_motor.mutate(to_mutate)
                    state1, state2 = gp_motor.crossover(prestate1, prestate2)
                    newpool.append(state1)
                    newpool.append(state2)

            self.pool += newpool

            return self.pool



# -------------------------------------------------------------------------- #
def init_tolerance(target, voc):

    # initial guess of tolerance:
    n_var = target.target[1]
    number_of_points = target.target[2][0].size
    ranges = target.target[-2]
    multfactor = 0.5

    initialguess = 0

    if n_var == 1:
        initialguess = multfactor*number_of_points * ranges[0]
    if n_var == 2:
        initialguess = multfactor*number_of_points * ranges[0] * ranges[1]
    if n_var == 3:
        initialguess = multfactor*number_of_points * ranges[0] * ranges[1] * ranges[2]

    if config.usederivativecost:
        initialguess= initialguess*2

    if config.findtolerance:
        tolerance = game_env.calculatetolerance(initialguess, target, voc)
    else:
        tolerance = initialguess

    return tolerance

#----------------------------------------------
def init_everything_else(which_target):
    # init targets
    train_target = Target(which_target, 'train')
    test_target = Target(which_target, 'test')

    # init dictionnaries
    voc_with_a = Voc(train_target, 'A')
    voc_no_a = Voc(train_target, 'noA')
    print('working with: ', voc_no_a.numbers_to_formula_dict)
    print('and then with: ', voc_with_a.numbers_to_formula_dict)

    # useful
    sizea = len(voc_with_a.numbers_to_formula_dict)
    sizenoa = len(voc_no_a.numbers_to_formula_dict)
    diff = sizenoa - sizea

    poolsize = 1000
    delete_ar1_ratio = 0.3

    extend_ratio = 2
    p_mutate = 0.4
    p_cross = 0.8

    binl_no_a = voc_no_a.maximal_size # number of bins for length of an eq
    maxl_no_a = voc_no_a.maximal_size

    #add rd eqs at each iteration
    addrandom = True

    return poolsize, delete_ar1_ratio, extend_ratio, p_mutate, p_cross, binl_no_a, maxl_no_a, addrandom, train_target, test_target, voc_with_a, voc_no_a, diff


# -------------------------------------------------------------------------- #
def main():


    id = str(int(10000000 * time.time()))

    for u in range(0, 6):

        #init target
        which_target = u

        # init para
        poolsize, delete_ar1_ratio, extend_ratio, p_mutate, p_cross, binl_no_a, maxl_no_a, addrandom, train_target, test_target, voc_with_a, voc_no_a, diff\
            = init_everything_else(which_target)

        # init tol
        tolerance = init_tolerance(train_target, voc_no_a)

        # init csv file
        mytarget = train_target.mytarget

        if config.uselocal:
            filepath = './' + str(id) + 'result_pure_gp_csv_file.csv'
        else:
            filepath = '/home/user/results/' + str(id) + 'result_pure_gp_csv_file.csv'

        with open(filepath, mode='a') as myfile:
            writer = csv.writer(myfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            writer.writerow(['target number' + str(which_target), mytarget])
            writer.writerow(['succes interger', 'iter (integer)', 'n_eq', 'total time (min)'])
            writer.writerow('\n')
        myfile.close()

        for run in range(20):
            success = False
            prefix = str(int(10000000 * time.time()))

            #init all eqs seen so far
            alleqs = {}

            gp = GP(delete_ar1_ratio, p_mutate, p_cross, maxl_no_a, poolsize, train_target, tolerance, extend_ratio, voc_no_a, addrandom, pool = None)

            iter_no_a = 150
            for i in range(iter_no_a):

                #init pool of workers
                print('this is iteration', i, 'working with tolerance', tolerance)

                # this creates a pool of states or extends it before evaluation
                pool = gp.extend_pool()
                print('pool creation/extension done')

                # dont evaluate again an equation already seen
                pool_to_eval = []
                results = []

                for state in pool:
                    if str(state.reversepolish) not in alleqs:
                        pool_to_eval.append(state)
                    else:
                        results.append([alleqs[str(state.reversepolish)], state])

                print('how many states to eval : ', len(pool_to_eval))

                for state in pool_to_eval:
                    reward, scalar_numbers, alla = game_env.game_evaluate(state.reversepolish, state.formulas, tolerance, voc_no_a, train_target, 'train')
                    alleqs.update({str(state.reversepolish): reward})
                    results.append([reward, state])

                print('pool eval done')

                rank_pool = sorted(results, key=itemgetter(0), reverse=True)[0:poolsize]

                truncated_pool = []
                for x in rank_pool:
                    truncated_pool.append(x[1])

                best = rank_pool[0]

                # save results and print
                printer = printresults(test_target, voc_no_a)

                valreward = printer.saveresults(i, tolerance, which_target, best, alleqs, prefix)

                if valreward > 0.999:
                    print('early stopping')
                    success = True
                    with open(filepath, mode='a') as myfile:
                        writer = csv.writer(myfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        timespent = (time.time() - eval(prefix) / 10000000) / 60
                        writer.writerow([str(1), str(i), str(len(alleqs)), str(timespent)])
                    myfile.close()

                    break

                gp = GP(delete_ar1_ratio, p_mutate, p_cross, maxl_no_a, poolsize, train_target, tolerance, extend_ratio, voc_no_a,
                        addrandom, truncated_pool)

            if success == False:
                with open(filepath, mode='a') as myfile:
                    writer = csv.writer(myfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    timespent = (time.time() - eval(prefix) / 10000000) / 60
                    writer.writerow([str(0), str(iter_no_a), str(len(alleqs)), str(timespent)])
                myfile.close()

# -------------------------------------------------------------------------- #



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


    def saveresults(self, i, tolerance, which_target, best, alleqs, prefix):

        best_reward, best_state = best
        best_formula = best_state.formulas

        if np.isnan(best_reward) or np.isinf(best_reward):
            best_reward=-1

        evaluate = Evaluatefit(best_formula, self.voc, self.target, tolerance, 'test')
        evaluate.rename_formulas()

        validation_reward = evaluate.eval_reward([])
        if np.isnan(validation_reward) or np.isinf(validation_reward):
            validation_reward = -1

        useful_form = self.finalrename(best_formula, [])

        if best_reward == 1.0:
            validation_reward = 1.0

        timespent = time.time() - eval(prefix)/10000000

        if config.uselocal:
            filepath = './' + prefix + 'results_pure_GP_target_' + str(which_target) + '.txt'
        else:
            filepath = '/home/user/results/'+ prefix+ 'results_pure_GP_target_' + str(which_target) + '.txt'
        with open(filepath, 'a') as myfile:

            myfile.write('iteration ' + str(i) + ': we have seen ' + str(len(alleqs)) + ' different eqs')
            myfile.write("\n")
            myfile.write("\n")

            myfile.write('best reward: ' + str(int(10000 * best_reward) / 10000) + ' with validation reward: ' + str(
                validation_reward))
            myfile.write("\n")
            myfile.write("\n")

            myfile.write('best eq: ' + str(useful_form) + ' ' + str(best_formula))
            myfile.write("\n")
            myfile.write("\n")

            myfile.write('time spent (in secs):' + str(timespent))
            myfile.write("\n")
            myfile.write("\n")
            myfile.write("---------------=============================----------------")
            myfile.write("\n")

            myfile.close()

        return validation_reward



if __name__ == '__main__':

    noprint = False

    if noprint:
        import sys

        class writer(object):
            log = []

            def write(self, data):
                self.log.append(data)

        logger = writer()
        sys.stdout = logger
        sys.stderr = logger
    main()






