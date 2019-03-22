import os
from gp_qd_class import printresults, GP_QD
import pickle
import multiprocessing as mp
import config
from game_env import Game
import game_env
from Targets import Target, Voc
from State import State

# -------------------------------------------------------------------------- #
def init_grid(reinit_grid, u):

    file_path = '/home/user/QD_pool' + str(u)+ '.txt'
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
def init_tolerance(target, voc, maxL):

    # initial guess of tolerance:
    n_var = target.target[1]
    number_of_points = target.target[2][0].size
    ranges = target.target[-1]
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
        tolerance = game_env.calculatetolerance(initialguess, target, voc, maxL)
    else:
        tolerance = initialguess

    return tolerance

# -------------------------------------------------------------------------- #
def evalme(onestate):
    train_target, voc, state, tolerance, maxL = onestate[0], onestate[1], onestate[2], onestate[3], onestate[4]

    #run 1:
    results = []
    reward, scalar_numbers, alla = game_env.game_evaluate(state.reversepolish, state.formulas, tolerance, voc, train_target, 'train')
    results.append([reward, scalar_numbers, alla])

    if config.tworunsineval :
        # run 2:
        reward, scalar_numbers, alla = game_env.game_evaluate(state.reversepolish, state.formulas, tolerance, voc, train_target, 'train')
        results.append([reward, scalar_numbers, alla])

        if results[0][0] >= results[1][0]:
            reward, scalar_numbers, alla = results[0]
        else:
            reward, scalar_numbers, alla = results[1]

    L = len(state.reversepolish)

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

    return reward, state, alla, scalar_numbers, L, function_number, powernumber, trignumber, explognumber


# -------------------------------------------------------------------------- #
def exec(which_target, train_target, test_target, voc, iteration, tolerance, gp):

    # init all eqs seen so far
    alleqs = {}

    for i in range(iteration):
        print('')
        print('this is iteration', i, 'working with tolerance', tolerance)
        # this creates a pool of states or extends it before evaluation
        pool = gp.extend_pool(alleqs)
        print('pool creation/extension done')

        pool_to_eval = []
        for state in pool:
            pool_to_eval.append([train_target, voc, state, tolerance, gp.maximal_size])

        print('how many states to eval : ', len(pool_to_eval))

        # init parallel workers
        mp_pool = mp.Pool(config.cpus)
        asyncResult = mp_pool.map_async(evalme, pool_to_eval)
        results = asyncResult.get()
        # close it
        mp_pool.close()
        mp_pool.join()

        print('pool eval done')

        for result in results:
            # this is for the fact that an equation that has already been seen might return a better reward, because cmaes method is not perfect!
            if str(result[1].reversepolish) in alleqs:
                if result[0] > alleqs[str(result[1].reversepolish)][0]:
                    alleqs.update({str(result[1].reversepolish): result})
            else:
                alleqs.update({str(result[1].reversepolish): result})

        results_by_bin = gp.bin_pool(results)

        # init
        if gp.QD_pool is None:
            gp.QD_pool = results_by_bin

        newbin, replacements = gp.update_qd_pool(results_by_bin)

        print('QD pool size', len(gp.QD_pool))
        print('alleqsseen', len(alleqs))

        # save results and print
        saveme = printresults(test_target, voc)
        valreward = saveme.saveresults(newbin, replacements, i, gp.QD_pool, gp.maxa, tolerance, which_target, alleqs)
        filename = '/home/user/QD_pool' + str(which_target) + '.txt'
        with open(filename, 'wb') as file:
            pickle.dump(gp.QD_pool, file)
        file.close()

        if valreward > 0.999:
            print('early stopping')
            return 'stop', gp.QD_pool

    return None, gp.QD_pool

# -----------------------------------------------#
def convert_eqs(qdpool, voc_a, voc_no_a, diff):
    # retrieve all the states, but replace integre scalars by generic scalar 'A' :
    allstates = []
    for binid in qdpool:
        state = qdpool[binid][1]

        newstate = []
        for char in state.reversepolish:
            if char == voc_no_a.terminalsymbol:
                newstate.append(voc_a.terminalsymbol)
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
        game = Game(voc_a, config.SENTENCELENGHT, creastate)
        game.simplif_eq()
        if voc_a.infinite_number not in game.state.reversepolish:
            initpool.append(game.state)

    return initpool

# -----------------------------------------------#
def eval_previous_eqs(which_target, train_target, test_target, voc_a, tolerance, initpool, gp):

    # init all eqs seen so far
    alleqs = {}

    pool_to_eval = []

    for state in initpool:
        pool_to_eval.append([train_target, voc_a, state, tolerance, gp.maximal_size])

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

    print('QD pool size', len(gp.QD_pool))
    print('alleqsseen', len(alleqs))

    # save results and print
    saveme = printresults(test_target, voc_a)
    valreward = saveme.saveresults(newbin, replacements, -1, gp.QD_pool, gp.maxa, tolerance, which_target, alleqs)
    filename = '/home/user/QD_pool' + str(which_target) + '.txt'
    with open(filename, 'wb') as file:
        pickle.dump(gp.QD_pool, file)
    file.close()

    if valreward > 0.999:
        print('early stopping')
        return alleqs, gp.QD_pool, 'stop'

    else:
        return alleqs, gp.QD_pool, None

# -----------------------------------------------#
def init_everything_else(which_target):
    # initial pool size of rd eqs at iteration 0
    poolsize = 3200
    # probability of drooping a function : cos(x) -> x
    delete_ar1_ratio = 0.3

    # pool extension by mutation and crossovers
    extend_ratio = 2

    # probabilities of mutation = p_mutate, crossovers
    p_mutate = 0.4
    # probabilities of crossovers = p_cross - p_mutate
    p_cross = 0.8
    # probability of mutate and cross = 1 - p_cross

    # using crossover the resulting eq can get super large : cap it to :
    maximal_size = config.SENTENCELENGHT

    #QD grid parameters
    bina = 20  # number of bins for number of free scalars
    maxa = 20
    binl = config.SENTENCELENGHT # number of bins for length of an eq
    maxl = config.SENTENCELENGHT
    binf = 8 # number of bins for number of fonctions
    maxf = 8
    new = 1
    binp = new  # number of bins for number of powers
    maxp = new
    bintrig = new # number of bins for number of trigonometric functions (sine and cos)
    maxtrig = new
    binexp = new # number of bins for number of exp-functions (exp or log)
    maxexp = new

    #add rd eqs at each iteration
    addrandom = True

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

    # init file of results
    if os.path.exists('results_target_' + str(which_target) + '.txt'):
        os.remove('results_target_' + str(which_target) + '.txt')

    return poolsize, delete_ar1_ratio, extend_ratio, p_mutate, p_cross, maximal_size, bina, maxa, binl, maxl, binf, maxf, \
           binp, maxp, bintrig, maxtrig, binexp, maxexp, addrandom, train_target, test_target, voc_with_a, voc_no_a, diff

# -----------------------------------------------#
def main():

    for u in range(0,4):

        # init target, dictionnaries, and meta parameters
        which_target = u
        poolsize, delete_ar1_ratio, extend_ratio, p_mutate, p_cross, maximal_size, bina, maxa, binl, maxl, binf, maxf, \
        binp, maxp, bintrig, maxtrig, binexp, maxexp, addrandom, train_target, test_target, voc_with_a, voc_no_a, diff = init_everything_else(which_target)

        # init qd grid
        reinit_grid = True
        qdpool = init_grid(reinit_grid, which_target)

        # ------------------- step 1 -----------------------#
        # first make a fast run with integers scalars to generate good equations for starting
        # the real run with free scalars 'A' precompute without 'A' : use it as initial grid

        #init tolerance :
        tolerance = init_tolerance(train_target, voc_no_a, maximal_size)

        #init gp class:
        gp = GP_QD(which_target, delete_ar1_ratio, p_mutate, p_cross, maximal_size, poolsize, train_target, voc_no_a, tolerance,
                   extend_ratio, maxa, bina, maxl, binl, maxf, binf, maxp, binp, maxtrig, bintrig, maxexp, binexp,
                   addrandom, None, None)

        # run evolution :
        iteration_no_a = 5
        stop, qdpool = exec(which_target, train_target, test_target, voc_no_a, iteration_no_a, tolerance, gp)
        # ------------------- step 2 -----------------------#
        #if target has not already been found, stop is None; then launch evoltion with free scalars A:
        if stop is None:
            # re-adjust tolerance in the case where free scalars are allowed :
            tolerance = init_tolerance(train_target, voc_with_a, maximal_size)
            # convert noA eqs into A eqs:
            initpool = convert_eqs(qdpool, voc_with_a, voc_no_a, diff)

            # reinit gp class with a:
            gp = GP_QD(which_target, delete_ar1_ratio, p_mutate, p_cross, maximal_size, poolsize, train_target,
                       voc_with_a, tolerance, extend_ratio, maxa, bina, maxl, binl, maxf, binf, maxp, binp, maxtrig, bintrig, maxexp, binexp,
                       addrandom, None, initpool)
            alleqs, QD_pool, stop = eval_previous_eqs(which_target, train_target, test_target, voc_with_a, tolerance, initpool, gp)

            # this might directly provide the exact solution : if not, stop is None, and thus, run evolution
            if stop is None:
                gp.QD_pool = QD_pool
                iteration_a = 50
                exec(which_target, train_target, test_target, voc_with_a, iteration_a, tolerance, gp)


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
