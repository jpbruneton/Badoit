import os
from gp_qd_class import printresults, GP_QD
import pickle
import multiprocessing as mp
import config
from game_env import Game
import game_env
from Targets import Target, Voc
from State import State
import time
import csv
import sys
import numpy as np
from Evaluate_fit import Evaluatefit
import pickle
# -------------------------------------------------------------------------- #
def init_grid(reinit_grid, type):

    file_path = 'QD_pool' + type +'.txt'

    if reinit_grid:
        if os.path.exists(file_path):
            os.remove(file_path)

    if os.path.exists(file_path) and config.saveqd:
        print('loading already trained model')

        with open(file_path, 'rb') as file:
            qdpool = pickle.load(file)
            file.close()

    else:
        print('grid doesnt exist')
        qdpool = None

    return qdpool

############
def save_qd_pool(pool, type):
    timeur = int(time.time()*1000000)
    if config.uselocal:
        file_path = 'QD_pool' + type + '.txt'
    else:
        file_path = '/home/user/results/QD_pool' + type + str(timeur) + '.txt'

    with open(file_path, 'wb') as file:
        pickle.dump(pool, file)
        file.close()

# -------------------------------------------------------------------------- #
def init_tolerance(target, voc):

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

# -------------------------------------------------------------------------- #
def evalme(onestate):
    test_target, voc, state, tolerance = onestate[0], onestate[1], onestate[2], onestate[3]

    results = []
    _, scalar_numbers, alla, rms = game_env.game_evaluate(state.reversepolish, state.formulas, tolerance, voc, test_target, 'test')
    results.append([rms, scalar_numbers, alla])

    if config.tworunsineval and voc.modescalar == 'A':
        # run 2:
        _, scalar_numbers, alla, rms = game_env.game_evaluate(state.reversepolish, state.formulas, tolerance, voc, test_target, 'test')
        results.append([rms, scalar_numbers, alla])

        if results[0][0] <= results[1][0]:
            rms, scalar_numbers, alla = results[0]
        else:
            rms, scalar_numbers, alla = results[1]

    if state.reversepolish[-1] == voc.terminalsymbol:
        L = len(state.reversepolish) -1
    else:
        L = len(state.reversepolish)

    if voc.modescalar == 'noA':
        scalar_numbers =0
        for char in voc.pure_numbers :
            scalar_numbers += state.reversepolish.count(char)

        scalar_numbers += state.reversepolish.count(voc.neutral_element)

    function_number = 0
    for char in voc.arity1symbols:
        function_number += state.reversepolish.count(char)

    powernumber = 0
    for char in state.reversepolish:
        if char == voc.power_number:
            powernumber += 1

    trignumber = 0
    for char in state.reversepolish:
        if char in voc.trignumbers:
            trignumber += 1

    explognumber = 0
    for char in state.reversepolish:
        if char in voc.explognumbers:
            explognumber += 1

    fnumber, deronenumber = 0, 0
    for char in state.reversepolish:
        if voc.modescalar == 'noA':
            if char == 6:
                fnumber = 1
            if char == 5:
                deronenumber = 1
        else:
            if char == 5:
                fnumber = 1
            if char == 4:
                deronenumber = 1

    return rms, state, alla, scalar_numbers, L, function_number, powernumber, trignumber, explognumber, fnumber, deronenumber


def count_meta_features(voc, state):
    if state.reversepolish[-1] == voc.terminalsymbol:
        L = len(state.reversepolish) -1
    else:
        L = len(state.reversepolish)

    scalar_numbers = 0
    for char in voc.pure_numbers:
        scalar_numbers += state.reversepolish.count(char)

    scalar_numbers += state.reversepolish.count(voc.neutral_element)
    scalar_numbers += state.reversepolish.count(voc.true_zero_number)

    function_number = 0
    for char in voc.arity1symbols:
        function_number += state.reversepolish.count(char)

    powernumber = 0
    for char in state.reversepolish:
        if char == voc.power_number:
            powernumber += 1

    trignumber = 0
    for char in state.reversepolish:
        if char in voc.trignumbers:
            trignumber += 1

    explognumber = 0
    for char in state.reversepolish:
        if char in voc.explognumbers:
            explognumber += 1

    fnumber, deronenumber = 0, 0
    for char in state.reversepolish:
        if voc.modescalar == 'noA':
            if char == 6:
                fnumber=1
            if char == 5:
                deronenumber = 1
        else:
            if char == 5:
                fnumber=1
            if char == 4:
                deronenumber = 1

    return scalar_numbers, L, function_number, powernumber, trignumber, explognumber, fnumber, deronenumber

# -------------------------------------------------------------------------- #
def exec(which_target, train_target, test_target, voc, iteration, tolerance, gp, prefix):

    local_alleqs = {}
    monocore = True
    for i in range(iteration):
        #parallel cma
        print('')
        print('this is iteration', i, 'working with tolerance', tolerance)
        # this creates a pool of states or extends it before evaluation
        pool = gp.extend_pool()
        print('pool creation/extension done')

        if voc.modescalar == 'A':
            print('mais ca passe ici?')
            pool_to_eval = []
            print('verif', len(pool))
            for state in pool:
                pool_to_eval.append([test_target, voc, state, tolerance])

            mp_pool = mp.Pool(config.cpus)
            asyncResult = mp_pool.map_async(evalme, pool_to_eval)
            results = asyncResult.get()
            # close it
            mp_pool.close()
            mp_pool.join()
            print('pool eval done')

            for result in results:
                # this is for the fact that an equation that has already been seen might return a better reward, because cmaes method is not perfect!
                if str(result[1].reversepolish) in local_alleqs:
                    if result[0] < local_alleqs[str(result[1].reversepolish)][0]:
                        local_alleqs.update({str(result[1].reversepolish): result})
                else:
                    local_alleqs.update({str(result[1].reversepolish): result})

        elif monocore == False:
            print('par pool')
            pool_to_eval = []
            print('verif', len(pool))
            for state in pool:
                pool_to_eval.append([test_target, voc, state, tolerance])

            mp_pool = mp.Pool(config.cpus)
            asyncResult = mp_pool.map_async(evalme, pool_to_eval)
            results = asyncResult.get()
            # close it
            mp_pool.close()
            mp_pool.join()
            print('pool eval done')

            for result in results:
                # this is for the fact that an equation that has already been seen might return a better reward, because cmaes method is not perfect!


                if str(result[1].reversepolish) in local_alleqs:
                    if result[0] < local_alleqs[str(result[1].reversepolish)][0]:
                        local_alleqs.update({str(result[1].reversepolish): result})
                else:
                    local_alleqs.update({str(result[1].reversepolish): result})
        else:
            results = []
            for state in pool:
                _, scalar_numbers, alla, rms = game_env.game_evaluate(state.reversepolish, state.formulas, tolerance, voc, test_target, 'test')
                evaluate = Evaluatefit(state.formulas, voc, test_target, tolerance, 'test')
                evaluate.rename_formulas()
                rms = evaluate.eval_reward_nrmse(alla)
                scalar_numbers, L, function_number, powernumber, trignumber, explognumber,  fnumber, deronenumber = count_meta_features(voc, state)
                results.append([rms, state, alla, scalar_numbers, L, function_number, powernumber, trignumber, explognumber, fnumber, deronenumber])
                if str(state.reversepolish) not in local_alleqs:
                    local_alleqs.update({str(state.reversepolish): results[-1]})

        results_by_bin = gp.bin_pool(results)

        # init
        if gp.QD_pool is None:
            gp.QD_pool = results_by_bin

        if voc.modescalar == 'A':
            type = '_a_'
        else:
            type = '_no_a_'


        newbin, replacements = gp.update_qd_pool(results_by_bin)

        save_qd_pool(gp.QD_pool, type)

        print('QD pool size', len(gp.QD_pool))
        print('alleqsseen', len(local_alleqs))


        # save results and print
        saveme = printresults(test_target, voc)
        valreward, valrmse, bf = saveme.saveresults(newbin, replacements, i, gp.QD_pool, gp.maxa, tolerance, which_target, local_alleqs, prefix)

        if valrmse <0.00000000001:
            del results
            print('early stopping')
            return 'es', gp.QD_pool, local_alleqs, i, valrmse
        if len(local_alleqs) > 10000000:
            del results
            print(' stopping bcs too many eqs seen')
            return 'stop', gp.QD_pool, local_alleqs, i, valrmse

    return None, gp.QD_pool, local_alleqs, i, valrmse, bf

# -----------------------------------------------#
def convert_eqs(qdpool, voc_a, voc_no_a, diff):
    # retrieve all the states, but replace integre scalars by generic scalar 'A' :
    allstates = []
    local_alleqs = {}
    for binid in qdpool:
        state = qdpool[binid][1]

        newstate = []
        for char in state.reversepolish:
            if char == voc_no_a.terminalsymbol:
                newstate.append(voc_a.terminalsymbol)
            elif char == voc_no_a.neutral_element:
                newstate.append(voc_a.pure_numbers[0])
            elif char == voc_no_a.true_zero_number:
                newstate.append(voc_a.pure_numbers[0])
            elif char in voc_no_a.pure_numbers:
                newstate.append(voc_a.pure_numbers[0])
            else:
                # shift everything : warning, works only if pure numbers are in the beginning of our dictionnaries!
                newstate.append(char - diff)

        allstates.append(newstate)

    # create the initial pool; now states have free scalars 'A'
    initpool = []

    # first simplfy the states, and remove infinities:
    for state in allstates:
        creastate = State(voc_a, state)
        if config.use_simplif:
            creastate = game_env.simplif_eq(voc_a, creastate)

        if voc_a.infinite_number not in creastate.reversepolish:
            if str(creastate.reversepolish) not in local_alleqs:
                local_alleqs.update({str(creastate.reversepolish): 1})
                initpool.append(creastate)

    del local_alleqs
    del allstates

    return initpool

# -----------------------------------------------#
def eval_previous_eqs(which_target, train_target, test_target, voc_a, tolerance, initpool, gp, prefix):

    # init all eqs seen so far
    alleqs = {}

    pool_to_eval = []
    for state in initpool:
        pool_to_eval.append([test_target, voc_a, state, tolerance])

    mp_pool = mp.Pool(config.cpus)
    print('how many states to eval : ', len(pool_to_eval))
    asyncResult = mp_pool.map_async(evalme, pool_to_eval)
    results = asyncResult.get()
    mp_pool.close()
    mp_pool.join()

    for result in results:
        alleqs.update({str(result[1].reversepolish): result})

    # bin the results
    results_by_bin = gp.bin_pool(results)
    gp.QD_pool = results_by_bin

    newbin, replacements = gp.update_qd_pool(results_by_bin)

    print('QD pool size', gp.QD_pool)
    print('alleqsseen', alleqs)

    # save results and print
    saveme = printresults(test_target, voc_a)
    valreward, valrmse, bf = saveme.saveresults(newbin, replacements, -1, gp.QD_pool, gp.maxa, tolerance, which_target, alleqs, prefix)

    del mp_pool
    del asyncResult
    del results

    if valrmse < 0.00000000001:
        print('early stopping')
        return alleqs, gp.QD_pool, 'stop', valrmse

    else:
        return alleqs, gp.QD_pool, None, valrmse, bf

# -----------------------------------------------#
def init_everything_else(which_target, maxsize):
    # init targets
    if not config.fromfile:
        train_target = Target(which_target, maxsize, 'train')
        test_target = Target(which_target, maxsize, 'test')
    else:
        train_target = Target(which_target,maxsize, 'train', 'fig1-waveform-H.txt')
        test_target = Target(which_target,maxsize, 'test', 'fig1-waveform-H.txt')

    # init dictionnaries
    voc_with_a = Voc(train_target, 'A')
    voc_no_a = Voc(train_target, 'noA')
    print('working with: ', voc_no_a.numbers_to_formula_dict)
    print('and then with: ', voc_with_a.numbers_to_formula_dict)

    # useful
    sizea = len(voc_with_a.numbers_to_formula_dict)
    sizenoa = len(voc_no_a.numbers_to_formula_dict)
    diff = sizenoa - sizea
    poolsize = 4000
    delete_ar1_ratio = 0.3
    extend_ratio = 2
    p_mutate = 0.4
    p_cross = 0.8

    binl_no_a = voc_no_a.maximal_size # number of bins for length of an eq
    maxl_no_a = voc_no_a.maximal_size
    bina = maxl_no_a  # number of bins for number of free scalars
    maxa = bina
    binl_a = voc_with_a.maximal_size # number of bins for length of an eq
    maxl_a = voc_with_a.maximal_size
    binf = 8 # number of bins for number of fonctions
    maxf = 8
    new = 0
    binp = new  # number of bins for number of powers
    maxp = new
    bintrig = new # number of bins for number of trigonometric functions (sine and cos)
    maxtrig = new
    binexp = new # number of bins for number of exp-functions (exp or log)
    maxexp = new
    derzero, derone = 1 , 1 #absence ou presence de fo et ou de fo'
    addrandom = False

    return poolsize, delete_ar1_ratio, extend_ratio, p_mutate, p_cross, bina, maxa, binl_no_a, maxl_no_a, binl_a, maxl_a, binf, maxf, \
           binp, maxp, bintrig, derzero, derone, maxtrig, binexp, maxexp, addrandom, train_target, test_target, voc_with_a, voc_no_a, diff

# -----------------------------------------------#
def main():
    id = str(int(10000000 * time.time()))

    for maxsize in [40]:#, 35, 40, 45]:
        best_formulas_of_this_size = []

        for u in range(1):
            # init target, dictionnaries, and meta parameters
            which_target = u
            poolsize, delete_ar1_ratio, extend_ratio, p_mutate, p_cross, bina, maxa,  binl_no_a, maxl_no_a, binl_a, maxl_a, binf, maxf, \
            binp, maxp, bintrig,  derzero, derone,  maxtrig, binexp, maxexp, addrandom, train_target, test_target, voc_with_a, voc_no_a, diff = init_everything_else(
                which_target, maxsize)
            #init csv file4000
            mytarget = train_target.mytarget
            if config.uselocal:
                filepath = './localdata/' + str(id)+'resultcsv_file.csv'
            else:
                filepath = '/home/user/results/' + str(id)+'resultcsv_file.csv'
            with open(filepath, mode='a') as myfile:
                writer = csv.writer(myfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

                writer.writerow(['target number' + str(which_target), mytarget])
                writer.writerow(['succes interger', 'iter (integer)', 'n_eq', 'succes_cma', 'n_eq', 'iter_cma',
                                 'total time (min)'])
                writer.writerow('\n')
            myfile.close()

            # init qd grid
            reinit_grid = False
            qdpool = init_grid(reinit_grid, '_no_a_')

            # ------------------- step 1 -----------------------#
            # first make a fast run with integers scalars to generate good equations for starting
            # the real run with free scalars 'A' precompute without 'A' : use it as initial grid

            #init tolerance :
            tolerance = init_tolerance(test_target, voc_no_a)

            #init gp class:
            gp = GP_QD(which_target, delete_ar1_ratio, p_mutate, p_cross, poolsize, voc_no_a, tolerance,
                       extend_ratio, maxa, bina, maxl_no_a, binl_no_a, maxf, binf, maxp, binp, maxtrig, bintrig,  derzero, derone, maxexp, binexp,
                       addrandom, qdpool, None)

            # trick for unique id of results file
            prefix = str(int(10000000 * time.time()))

            if False:
                formm = '((f0)/(((np.log(x0))+(0.742001))+(((-2.709957)+(x0))*(np.cos((2.045489)-(x0))))))/(0.004552)'
                evall = Evaluatefit(formm, voc_with_a, test_target, tolerance, 'test')
                print('re', evall.formulas)

                evall.rename_formulas()
                print(evall.formulas)
                rew, scalar_numbers, alla, rms = game_env.game_evaluate([1], evall.formulas, tolerance, voc_no_a, test_target, 'test')
                print('donne:', rew, scalar_numbers, alla, rms)
                time.sleep(500)
            # run evolution :
            iteration_no_a = 1
            stop, qdpool, alleqs_no_a, iter_no_a, valrmse, bf = exec(which_target, train_target, test_target, voc_no_a, iteration_no_a, tolerance, gp, prefix)


            if stop is not None:
                if stop == 'es':
                    with open(filepath, mode='a') as myfile:
                        writer = csv.writer(myfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        timespent = (time.time() - eval(prefix) / 10000000)/60
                        writer.writerow([str(1), str(iter_no_a), str(len(alleqs_no_a)), '0', '0', '0', str(valrmse), str(timespent)])
                    myfile.close()
                else:
                    with open(filepath, mode='a') as myfile:
                        writer = csv.writer(myfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        timespent = (time.time() - eval(prefix) / 10000000)/60
                        writer.writerow([str(0), str(iter_no_a), str(len(alleqs_no_a)), '0', '0', '0', str(valrmse), str(timespent)])
                    myfile.close()


            # ------------------- step 2 -----------------------#
            #if target has not already been found, stop is None; then launch evoltion with free scalars A:
            if stop is None:

                # re-adjust tolerance in the case where free scalars are allowed :
                tolerance = init_tolerance(train_target, voc_with_a)
                # convert noA eqs into A eqs:
                if config.saveqd:
                    initpool = init_grid(False, '_a_')
                    QD_pool = initpool
                else:
                    initpool = convert_eqs(qdpool, voc_with_a, voc_no_a, diff)

                # reinit gp class with a:
                gp = GP_QD(which_target, delete_ar1_ratio, p_mutate, p_cross, poolsize,
                           voc_with_a, tolerance, extend_ratio, maxa, bina, maxl_a, binl_a, maxf, binf, maxp, binp, maxtrig, bintrig,  derzero, derone, maxexp, binexp,
                           addrandom, initpool, None)
                if not config.saveqd:
                    alleqs_change_mode, QD_pool, stop, valrmse, bf = eval_previous_eqs(which_target, train_target, test_target, voc_with_a, tolerance, initpool, gp, prefix)
                else:
                    stop = None

                if stop is not None:
                    with open(filepath, mode='a') as myfile:
                        writer = csv.writer(myfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        timespent = (time.time() - eval(prefix) / 10000000) / 60
                        writer.writerow([str(0), str(iter_no_a), str(len(alleqs_no_a)), '1', str(len(alleqs_change_mode)), '0',  str(valrmse), str(timespent)])
                    myfile.close()

                # this might directly provide the exact solution : if not, stop is None, and thus, run evolution
                if stop is None:
                    gp.QD_pool = QD_pool
                    iteration_a = 50

                    stop, qdpool, alleqs_a, iter_a, valrmse, bf = exec(which_target, train_target, test_target, voc_with_a, iteration_a, tolerance, gp, prefix)

                    if stop is None:
                        success = 0
                    else:
                        success = 1

                    with open(filepath, mode='a') as myfile:
                        writer = csv.writer(myfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        timespent = (time.time() - eval(prefix) / 10000000) / 60
                        writer.writerow(
                            [str(0), str(iter_no_a), str(len(alleqs_no_a)), str(success), str(len(alleqs_a)), str(iter_a+1),  str(valrmse), str(timespent)])
                    myfile.close()
            best_formulas_of_this_size.append(bf)

# -----------------------------------------------#
if __name__ == '__main__':

    # don't display any output
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
    #     print(newgame.state.formulas)
    #     #newgame.simplif_eq()
    #     #print(newgame.state.formulas)
    #     game_env.simplif_eq(voc_no_a, newgame.state)

