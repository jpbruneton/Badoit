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

# -------------------------------------------------------------------------- #
def init_grid(reinit_grid, u):
    if config.uselocal:
        file_path = './gpdata/QD_pool' + str(u) + '.txt'
    else:
        file_path = '/home/user/QD_pool' + str(u) + '.txt'

    if reinit_grid:
        if os.path.exists(file_path):
            os.remove(file_path)

    if os.path.exists(file_path):
        print('loading already trained model')

        with open(file_path, 'rb') as file:
            qdpool = pickle.load(file)
            file.close()
    else:
        print('grid doesnt exist')
        qdpool = None

    return qdpool

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

# -------------------------------------------------------------------------- #
def evalme(onestate):
    train_target, voc, state, tolerance = onestate[0], onestate[1], onestate[2], onestate[3]
    #failurestate = State(voc, [voc.neutral_element,1])

    #print(state.reversepolish, state.formulas)
    #if len(state.reversepolish) == 0 or len(state.reversepolish) == 1:
        #print('oui', state.reversepolish, state.formulas)
    #    return -1, failurestate, [], 0, 2, 0, 0, 0, 0


    #run 1:
    results = []
    #try:
    reward, scalar_numbers, alla = game_env.game_evaluate(state.reversepolish, state.formulas, tolerance, voc, train_target, 'train')
    results.append([reward, scalar_numbers, alla])
    #except (ValueError, IndexError, AttributeError, RuntimeError, RuntimeWarning):
    #    print('happening')
    #    reward, state, scalar_numbers, alla = -1, failurestate, 0, []

    if config.tworunsineval and voc.modescalar == 'A':
        # run 2:
        reward, scalar_numbers, alla = game_env.game_evaluate(state.reversepolish, state.formulas, tolerance, voc, train_target, 'train')
        results.append([reward, scalar_numbers, alla])

        if results[0][0] >= results[1][0]:
            reward, scalar_numbers, alla = results[0]
        else:
            reward, scalar_numbers, alla = results[1]


    #try:
        # print('y', L)
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
    #print('t',state.reversepolish, state.formulas, 'k', reward, state, alla, scalar_numbers, L, function_number, powernumber, trignumber, explognumber)
    return reward, state, alla, scalar_numbers, L, function_number, powernumber, trignumber, explognumber


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

    return scalar_numbers, L, function_number, powernumber, trignumber, explognumber
    #
    # except (ValueError, IndexError, AttributeError, RuntimeError, RuntimeWarning):
    #     #print(state.formulas, state.reversepolish)
    #     print('happening')
    #     if config.uselocal:
    #         filepath = './bureport.txt'
    #     else:
    #         filepath = '/home/user/results/bureport.txt'
    #     with open(filepath, 'a') as myfile:
    #         myfile.write(str(state.formulas) + str(state.reversepolish))
    #     myfile.close()
    #     return -1, failurestate, [], 0, 2, 0, 0, 0, 0

# -------------------------------------------------------------------------- #
def exec(which_target, train_target, test_target, voc, iteration, tolerance, gp, prefix):

    # init all eqs seen so far
    #mp_pool = mp.Pool(config.cpus)
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
            pool_to_eval = []
            print('verif', len(pool))
            for state in pool:
                pool_to_eval.append([train_target, voc, state, tolerance])

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
                    if result[0] > local_alleqs[str(result[1].reversepolish)][0]:
                        local_alleqs.update({str(result[1].reversepolish): result})
                else:
                    local_alleqs.update({str(result[1].reversepolish): result})

        elif monocore == False:
            print('par pool')
            pool_to_eval = []
            print('verif', len(pool))
            for state in pool:
                pool_to_eval.append([train_target, voc, state, tolerance])

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
                    if result[0] > local_alleqs[str(result[1].reversepolish)][0]:
                        local_alleqs.update({str(result[1].reversepolish): result})
                else:
                    local_alleqs.update({str(result[1].reversepolish): result})


        else:
            #mono core eval
            #print('monocore eval')
            results = []
            #print('lenpool', len(pool))
            for state in pool:
                reward, scalar_numbers, alla = game_env.game_evaluate(state.reversepolish, state.formulas, tolerance, voc, train_target, 'train')
                scalar_numbers, L, function_number, powernumber, trignumber, explognumber = count_meta_features(voc, state)
                #print(state.formulas)
                #print(L, function_number, powernumber, trignumber, explognumber)
                results.append([reward, state, alla, scalar_numbers, L, function_number, powernumber, trignumber, explognumber])
                if str(state.reversepolish) not in local_alleqs:
                    local_alleqs.update({str(state.reversepolish): results[-1]})
            #print('yo', len(results))
        results_by_bin = gp.bin_pool(results)

        # init
        if gp.QD_pool is None:
            gp.QD_pool = results_by_bin

        newbin, replacements = gp.update_qd_pool(results_by_bin)

        print('QD pool size', len(gp.QD_pool))
        print('alleqsseen', len(local_alleqs))

        # save results and print
        saveme = printresults(test_target, voc)
        valreward, valrmse = saveme.saveresults(newbin, replacements, i, gp.QD_pool, gp.maxa, tolerance, which_target, local_alleqs, prefix)
    #    if config.uselocal:
     #       filename = './gpdata/QD_pool'+ str(which_target) + '.txt'
     #   else:
     #       filename = '/home/user/QD_pool' + str(which_target) + '.txt'
     #   with open(filename, 'wb') as file:
     #       pickle.dump(gp.QD_pool, file)
     #   file.close()

        if valreward > 0.999:
            #mp_pool.close()
            #mp_pool.join()
     #       del mp_pool
      #      del asyncResult
            del results
            print('early stopping')
            return 'stop', gp.QD_pool, local_alleqs, i, valrmse
    #mp_pool.close()
    #mp_pool.join()
   # del mp_pool
    #del asyncResult
    #del results
    return None, gp.QD_pool, local_alleqs, i, valrmse

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
        pool_to_eval.append([train_target, voc_a, state, tolerance])

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
    valreward, valrmse = saveme.saveresults(newbin, replacements, -1, gp.QD_pool, gp.maxa, tolerance, which_target, alleqs, prefix)
    # if config.uselocal:
    #     filename = './gpdata/QD_pool' + str(which_target) + '.txt'
    # else:
    #     filename = '/home/user/QD_pool' + str(which_target) + '.txt'
    # with open(filename, 'wb') as file:
    #     pickle.dump(gp.QD_pool, file)
    # file.close()

    del mp_pool
    del asyncResult
    del results

    if valreward > 0.999:
        print('early stopping')
        return alleqs, gp.QD_pool, 'stop', valrmse

    else:
        return alleqs, gp.QD_pool, None, valrmse

# -----------------------------------------------#
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

    # initial pool size of rd eqs at iteration 0
    poolsize = 4000
    # probability of dropping a function : cos(x) -> x
    delete_ar1_ratio = 0.3

    # pool extension by mutation and crossovers
    extend_ratio = 2.3

    # probabilities of mutation = p_mutate, crossovers
    p_mutate = 0.4
    # probabilities of crossovers = p_cross - p_mutate
    p_cross = 0.8
    # probability of mutate and cross = 1 - p_cross


    #QD grid parameters
    bina = 20  # number of bins for number of free scalars
    maxa = 20
    binl_no_a = voc_no_a.maximal_size # number of bins for length of an eq
    maxl_no_a = voc_no_a.maximal_size
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

    #add rd eqs at each iteration
    addrandom = True


    return poolsize, delete_ar1_ratio, extend_ratio, p_mutate, p_cross, bina, maxa, binl_no_a, maxl_no_a, binl_a, maxl_a, binf, maxf, \
           binp, maxp, bintrig, maxtrig, binexp, maxexp, addrandom, train_target, test_target, voc_with_a, voc_no_a, diff

# -----------------------------------------------#
def main():
    id = str(int(10000000 * time.time()))
    targetsafinir=[29]
    for u in targetsafinir:

        # init target, dictionnaries, and meta parameters
        which_target = u
        poolsize, delete_ar1_ratio, extend_ratio, p_mutate, p_cross, bina, maxa,  binl_no_a, maxl_no_a, binl_a, maxl_a, binf, maxf, \
        binp, maxp, bintrig, maxtrig, binexp, maxexp, addrandom, train_target, test_target, voc_with_a, voc_no_a, diff = init_everything_else(
            which_target)

        #init csv file
        mytarget = train_target.mytarget
        if config.uselocal:
            filepath = './' + str(id)+'resultcsv_file.csv'
        else:
            filepath = '/home/user/results/' + str(id)+'resultcsv_file.csv'
        with open(filepath, mode='a') as myfile:
            writer = csv.writer(myfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            writer.writerow(['target number' + str(which_target), mytarget])
            writer.writerow(['succes interger', 'iter (integer)', 'n_eq', 'succes_cma', 'n_eq', 'iter_cma',
                             'total time (min)'])
            writer.writerow('\n')
        myfile.close()

        for runs in range(1):

            # init qd grid
            reinit_grid = True
            qdpool = init_grid(reinit_grid, which_target)


            # ------------------- step 1 -----------------------#
            # first make a fast run with integers scalars to generate good equations for starting
            # the real run with free scalars 'A' precompute without 'A' : use it as initial grid

            #init tolerance :
            tolerance = init_tolerance(train_target, voc_no_a)

            #init gp class:
            gp = GP_QD(which_target, delete_ar1_ratio, p_mutate, p_cross, poolsize, voc_no_a, tolerance,
                       extend_ratio, maxa, bina, maxl_no_a, binl_no_a, maxf, binf, maxp, binp, maxtrig, bintrig, maxexp, binexp,
                       addrandom, None, None)

            # trick for unique id of results file
            prefix = str(int(10000000 * time.time()))

            # run evolution :
            iteration_no_a = 150
            stop, qdpool, alleqs_no_a, iter_no_a, valrmse = exec(which_target, train_target, test_target, voc_no_a, iteration_no_a, tolerance, gp, prefix)

            test = False
            if test == False:
                #save csv
                if stop is not None:
                    with open(filepath, mode='a') as myfile:
                        writer = csv.writer(myfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        timespent = (time.time() - eval(prefix) / 10000000)/60
                        writer.writerow([str(1), str(iter_no_a), str(len(alleqs_no_a)), '0', '0', '0', str(valrmse), str(timespent)])
                    myfile.close()


                # ------------------- step 2 -----------------------#
                #if target has not already been found, stop is None; then launch evoltion with free scalars A:
                if stop is None:

                    # re-adjust tolerance in the case where free scalars are allowed :
                    tolerance = init_tolerance(train_target, voc_with_a)
                    # convert noA eqs into A eqs:
                    initpool = convert_eqs(qdpool, voc_with_a, voc_no_a, diff)

                    # reinit gp class with a:
                    gp = GP_QD(which_target, delete_ar1_ratio, p_mutate, p_cross, poolsize,
                               voc_with_a, tolerance, extend_ratio, maxa, bina, maxl_a, binl_a, maxf, binf, maxp, binp, maxtrig, bintrig, maxexp, binexp,
                               addrandom, None, initpool)
                    alleqs_change_mode, QD_pool, stop, valrmse = eval_previous_eqs(which_target, train_target, test_target, voc_with_a, tolerance, initpool, gp, prefix)

                    if stop is not None:
                        with open(filepath, mode='a') as myfile:
                            writer = csv.writer(myfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                            timespent = (time.time() - eval(prefix) / 10000000) / 60
                            writer.writerow([str(0), str(iter_no_a), str(len(alleqs_no_a)), '1', str(len(alleqs_change_mode)), '0',  str(valrmse), str(timespent)])
                        myfile.close()

                    # this might directly provide the exact solution : if not, stop is None, and thus, run evolution
                    if stop is None:
                        gp.QD_pool = QD_pool
                        iteration_a = 150

                        stop, qdpool, alleqs_a, iter_a, valrmse = exec(which_target, train_target, test_target, voc_with_a, iteration_a, tolerance, gp, prefix)

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

            #del alleqs_change_mode
            #del alleqs_a
            #del alleqs_no_a
            #del gp
            #del initpool

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


    #main()
    test = False
    if test:
        which_target = 0
        poolsize, delete_ar1_ratio, extend_ratio, p_mutate, p_cross, bina, maxa, binl_no_a, maxl_no_a, binl_a, maxl_a, binf, maxf, \
        binp, maxp, bintrig, maxtrig, binexp, maxexp, addrandom, train_target, test_target, voc_with_a, voc_no_a, diff = init_everything_else(
             which_target)
        for u in range(10000):
            newgame = game_env.randomeqs(voc_no_a)
            if voc_no_a.infinite_number not in newgame.state.reversepolish:
                if newgame.getnumberoffunctions() > config.MAX_DEPTH:
                    print('happ')

    else:
        main()
    #     print(newgame.state.formulas)
    #     #newgame.simplif_eq()
    #     #print(newgame.state.formulas)
    #     game_env.simplif_eq(voc_no_a, newgame.state)

