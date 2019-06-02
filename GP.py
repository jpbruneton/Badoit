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
    def __init__(self, delete_ar1_ratio, p_mutate, p_cross, maximal_size, tournament, newpool, poolsize,  target, tolerance, extend_ratio, voc, addrandom, howmany, pool = None):

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
        self.tournament = tournament
        self.newpool = newpool
        self.howmany = howmany
    def par_crea(self, task):
        np.random.seed(task)
        newgame = game_env.randomeqs(self.voc)
        return newgame.state
    # ---------------------------------------------------------------------------- #
    # init a new pool of rd eqs if doesnt exists, or extend it by mutations/crossovers

    def extend_pool(self):

        if self.pool == None:
            self.pool = []
#            tasks = range(0, self.poolsize)

 #           mp_pool = mp.Pool(config.cpus)
  #          asyncResult = mp_pool.map_async(self.par_crea, tasks)
   #         results = asyncResult.get()
    #        mp_pool.close()
     #       mp_pool.join()
            for j in range(self.poolsize):
                newgame = game_env.randomeqs(self.voc)

                if self.voc.infinite_number not in newgame.state.reversepolish:
                    self.pool.append(newgame.state)

            return self.pool

        #extend the pool by a factor 2, say
        else:
            gp_motor = generate_offsprings(self.delete_ar1_ratio, self.p_mutate, self.p_cross, self.maximal_size, self.voc, self.maximal_size)

            all_states = []

            cutpool =[]
            for elem in self.pool:
                all_states.append(elem)

            if self.howmany < 200:
                c=0
                rdpool = []
                while c < int(self.poolsize/2):
                    newstate = game_env.randomeqs(self.voc).state
                    if self.voc.infinite_number not in newstate.reversepolish:
                        reward, scalar_numbers, alla = game_env.game_evaluate(newstate.reversepolish, newstate.formulas,self.tolerance, self.voc, self.target, 'train')
                        all_states.append([reward,newstate])
                        c += 1

            for elem in all_states:
                cutpool.append(elem[1])

            self.pool = cutpool

            print('on commence avec', len(self.pool))
            #self.pool = []

            print('sizetocross', len(all_states))

            #then mutate and crossover
            newpool = []
            count=0
            while len(newpool) < self.newpool and count < 40000:
                kelem=[]
                if self.howmany < 100:
                    for k in range(2):
                        index = np.random.randint(0, len(all_states))
                        # print(index)
                        kelem.append(all_states[index])
                else:
                    for k in range(self.tournament):
                        index = np.random.randint(0, len(all_states))
                        #print(index)
                        kelem.append(all_states[index])

                rank = sorted(kelem, key=itemgetter(0), reverse=True)
                first = copy.deepcopy(rank[0][1])
                second = copy.deepcopy(rank[1][1])

                #index = np.random.randint(0, len(all_states))
                #state = all_states[index]
                u = random.random()

                if u <= self.p_mutate:
                    count += 1
                    success, mutatedstate = gp_motor.mutate(first)
                    newpool.append(mutatedstate)

                elif u <= self.p_cross:
                    count += 2
                    #index = np.random.randint(0, len(all_states))
                    #otherstate = all_states[index]  # this might crossover with itself : why not!
                    success, state1, state2 = gp_motor.crossover(first, second)
                    if success:
                        newpool.append(state1)
                        newpool.append(state2)

                else:  # mutate AND cross
                    count += 2

                    #index = np.random.randint(0, len(all_states))
                    #to_mutate = copy.deepcopy(all_states[index])
                    s1, prestate1 = gp_motor.mutate(first)
                    s2, prestate2 = gp_motor.mutate(second)
                    success, state1, state2 = gp_motor.crossover(prestate1, prestate2)
                    if success:
                        newpool.append(state1)
                        newpool.append(state2)

            self.pool += newpool
            print('finally', len(self.pool), 'and count', count)
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
    tournament = 2
    newpool = 2000

    extend_ratio = 2
    p_mutate = 0.4
    p_cross = 0.8

    binl_no_a = voc_no_a.maximal_size # number of bins for length of an eq
    maxl_no_a = voc_no_a.maximal_size

    #add rd eqs at each iteration
    addrandom = False

    return tournament, newpool, poolsize, delete_ar1_ratio, extend_ratio, p_mutate, p_cross, binl_no_a, maxl_no_a, addrandom, train_target, test_target, voc_with_a, voc_no_a, diff


# -------------------------------------------------------------------------- #
def main(task):


    id = str(int(10000000 * time.time()))

    for u in range(5, 6):

        #init target
        which_target = u

        # init para
        tournament, newpool, poolsize, delete_ar1_ratio, extend_ratio, p_mutate, p_cross, binl_no_a, maxl_no_a, addrandom, train_target, test_target, voc_with_a, voc_no_a, diff\
            = init_everything_else(which_target)

        # init tol
        tolerance = init_tolerance(train_target, voc_no_a)

        # init csv file
        mytarget = train_target.mytarget

        if config.uselocal:
            filepath = './localdata/' + str(id) + 'result_pure_gp_csv_file.csv'
        else:
            filepath = '/home/user/results/' + str(id) + 'result_pure_gp_csv_file.csv'

        # with open(filepath, mode='a') as myfile:
        #     writer = csv.writer(myfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        #
        #     writer.writerow(['target number' + str(which_target), mytarget])
        #     writer.writerow(['succes interger', 'iter (integer)', 'n_eq', 'total time (min)'])
        #     writer.writerow('\n')
        # myfile.close()

        for run in range(1):
            success = False
            prefix = str(int(10000000 * time.time()))

            #init all eqs seen so far
            alleqs = {}

            gp = GP(delete_ar1_ratio, p_mutate, p_cross, maxl_no_a, tournament, newpool, poolsize, train_target, tolerance, extend_ratio, voc_no_a, addrandom, 1000, pool = None)

            iter_no_a = 1500
            for i in range(iter_no_a):

                #init pool of workers
                print('this is iteration', i, 'working with tolerance', tolerance)

                # this creates a pool of states or extends it before evaluation
                pool = gp.extend_pool()
                print('pool creation/extension done', len(pool))

               # if addrandom:
                #    for i in range(500):
                 #       newgame = game_env.randomeqs(voc_no_a)
                  #      pool.append(newgame.state)

                # dont evaluate again an equation already seen
                pool_to_eval = []
                results = []

                for state in pool:
                    if str(state.reversepolish) not in alleqs:
                        pool_to_eval.append(state)
                    else:
                        results.append([alleqs[str(state.reversepolish)], state])

                print('how many states to eval : ', len(pool_to_eval))
                howmany = len(pool_to_eval)
                for state in pool_to_eval:
                    reward, scalar_numbers, alla = game_env.game_evaluate(state.reversepolish, state.formulas, tolerance, voc_no_a, train_target, 'train')
                    alleqs.update({str(state.reversepolish): reward})
                    results.append([reward, state])

                print('pool eval done, alleqs : ', len(alleqs))

                rank_pool = sorted(results, key=itemgetter(0), reverse=True)[0:poolsize]

                truncated_pool = rank_pool
                #for x in rank_pool:
                #    truncated_pool.append([x[0],x[1]])

                best = rank_pool[0]

                # save results and print
                printer = printresults(test_target, voc_no_a)

                valreward = printer.saveresults(i, tolerance, which_target, best, alleqs, prefix)

                if valreward > 0.999:
                    print('early stopping')
                    success = True
                    # with open(filepath, mode='a') as myfile:
                    #     writer = csv.writer(myfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    timespent = (time.time() - eval(prefix) / 10000000) / 60
                    #     writer.writerow([str(1), str(i), str(len(alleqs)), str(timespent)])
                    # myfile.close()
                    return [str(1), str(i), str(len(alleqs)), str(timespent)]
                    break

                if len(alleqs) > 100000:

                    break

                gp = GP(delete_ar1_ratio, p_mutate, p_cross, maxl_no_a,  tournament, newpool, poolsize, train_target, tolerance, extend_ratio, voc_no_a,
                        addrandom, howmany, truncated_pool)

            if success == False:
                # with open(filepath, mode='a') as myfile:
                #     writer = csv.writer(myfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                timespent = (time.time() - eval(prefix) / 10000000) / 60
                #     writer.writerow([str(0), str(iter_no_a), str(len(alleqs)), str(timespent)])
                # myfile.close()
                return [str(0), str(iter_no_a), str(len(alleqs)), str(timespent)]


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
        #
        # if config.uselocal:
        #     filepath = './localdata/' + prefix + 'results_pure_GP_target_' + str(which_target) + '.txt'
        # else:
        #     filepath = '/home/user/results/'+ prefix+ 'results_pure_GP_target_' + str(which_target) + '.txt'
        # with open(filepath, 'a') as myfile:
        #
        #     myfile.write('iteration ' + str(i) + ': we have seen ' + str(len(alleqs)) + ' different eqs')
        #     myfile.write("\n")
        #     myfile.write("\n")
        #
        #     myfile.write('best reward: ' + str(int(10000 * best_reward) / 10000) + ' with validation reward: ' + str(
        #         validation_reward))
        #     myfile.write("\n")
        #     myfile.write("\n")
        #
        #     myfile.write('best eq: ' + str(useful_form) + ' ' + str(best_formula))
        #     myfile.write("\n")
        #     myfile.write("\n")
        #
        #     myfile.write('time spent (in secs):' + str(timespent))
        #     myfile.write("\n")
        #     myfile.write("\n")
        #     myfile.write("---------------=============================----------------")
        #     myfile.write("\n")
        #
        #     myfile.close()

        return validation_reward

def paramain():
    tasks = range(0, 100)

    mp_pool = mp.Pool(config.cpus)
    asyncResult = mp_pool.map_async(main, tasks)
    results = asyncResult.get()
    mp_pool.close()
    mp_pool.join()
    return results

import time
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
    allresults = paramain()
    print(allresults)
    if config.uselocal:
        filepath = './localdata/results_pure_GP_target_test' + str(5) + str(int(1000000*time.time())) + '.csv'

    with open(filepath, mode='w') as myfile:
        writer = csv.writer(myfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for elem in allresults:
            res=[]
            for x in elem:
                res.append(float(x))
            writer.writerow(res)
    myfile.close()





