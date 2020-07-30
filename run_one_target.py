#  ======================== CMA-Based Symbolic Regressor ========================== #
# Project:          Symbolic regression for physics
# Name:             AST.py
# Authors:          Jean-Philippe Bruneton
# Date:             2020
# License:          BSD 3-Clause License
# ============================================================================ #


# ================================= PREAMBLE ================================= #
# Packages
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



# -----------------------------------------------#
def convert_eqs(qdpool, voc_a, voc_no_a, diff, calculus_mode):
    # retrieve all the states, but replace integre scalars by generic scalar 'A' :
    allstates = []
    local_alleqs = {}
    for binid in qdpool:
        state = qdpool[binid][1]

        newstate = []
        for char in state.reversepolish:
            if char in voc_no_a.terminalsymbol:
                newstate.append(voc_a.terminalsymbol[0])
            elif char in voc_no_a.neutral_element:
                newstate.append(voc_a.pure_numbers[0])
            elif char in voc_no_a.true_zero_number:
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
        creastate = State(voc_a, state, calculus_mode)
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
def eval_previous_eqs(train_targets, voc_a, u, initpool, gp, look_for, calculus_mode, name, qdpoolname,sttime):

    # init all eqs seen so far
    alleqs = {}

    pool_to_eval = []
    for state in initpool:
        pool_to_eval.append([train_targets, voc_a, state, u, look_for])

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
    save_qd_pool(gp.QD_pool, qdpoolname)

    print('QD pool size', len(gp.QD_pool))
    print('alleqsseen', len(alleqs))

    # save results and print
    saveme = printresults(train_targets, voc_a, calculus_mode, look_for, name)
    valrmse, bf = saveme.saveresults(newbin, replacements, -1, gp.QD_pool, gp.bin_nscalar, alleqs, sttime, u, look_for)

    del mp_pool
    del asyncResult
    del results

    if valrmse < 0.00000000001:
        print('early stopping')
        return alleqs, gp.QD_pool, 'stop', valrmse, bf

    else:
        return alleqs, gp.QD_pool, None, valrmse, bf


# -------------------------------------------------------------------------- #
def init_grid(reinit_grid, poolname):

    if config.uselocal == False:
        filepath = '/home/user/results/' + poolname
    else:
        filepath  = 'results/' + poolname

    if reinit_grid and os.path.exists(filepath):
        os.remove(filepath)

    if os.path.exists(filepath) and config.loadexistingmodel:
        print('loading already trained model')

        with open(poolname, 'rb') as file:
            qdpool = pickle.load(file)
            file.close()

    else:
        print('grid doesnt exist')
        qdpool = None

    return qdpool

# -------------------------------------------------------------------------- #
def save_qd_pool(pool, qdpoolname):
    if config.uselocal:
        file_path = 'results/' + qdpoolname
    else:
        file_path = '/home/user/results/' + qdpoolname

    file = open(file_path, 'wb')
    pickle.dump(pool, file)
    file.close()

# -------------------------------------------------------------------------- #
def evalme(onestate):
    train_targets, voc, state, u, look_for = onestate[0], onestate[1], onestate[2], onestate[3], onestate[4]
    results = []
    scalar_numbers, alla, rms = game_env.game_evaluate(state.reversepolish, state.formulas, voc, train_targets, 'train',u, look_for)
    results.append([rms, scalar_numbers, alla])

    if config.tworunsineval:
        scalar_numbers, alla, rms = game_env.game_evaluate(state.reversepolish, state.formulas, voc, train_targets, 'train',u, look_for)
        results.append([rms, scalar_numbers, alla])
        if results[0][0] <= results[1][0]:
            rms, scalar_numbers, alla = results[0]
        else:
            rms, scalar_numbers, alla = results[1]

    return rms, state, alla, scalar_numbers

# -------------------------------------------------------------------------- #
def exec(train_targets, u, voc, iteration, gp, look_for, calculus_mode, name, qdpoolname, starttime):

    local_alleqs = {}
    for i in range(iteration):
        print('')
        print('this is iteration', i)
        pool = gp.extend_pool()

        pool_to_eval = []
        for state in pool:
            pool_to_eval.append([train_targets, voc, state, u, look_for])

        mp_pool = mp.Pool(config.cpus)
        asyncResult = mp_pool.map_async(evalme, pool_to_eval)
        results = asyncResult.get()
        mp_pool.close()
        mp_pool.join()

        for result in results:
            # this is for the fact that an equation that has already been seen might return a better reward,
            # because cmaes method is not perfect!
            if str(result[1].reversepolish) in local_alleqs:
                if result[0] < local_alleqs[str(result[1].reversepolish)][0]:
                    local_alleqs.update({str(result[1].reversepolish): result})
            else:
                local_alleqs.update({str(result[1].reversepolish): result})

        results_by_bin = gp.bin_pool(results)

        # init
        if gp.QD_pool is None:
            gp.QD_pool = results_by_bin

        newbin, replacements = gp.update_qd_pool(results_by_bin)
        save_qd_pool(gp.QD_pool, qdpoolname)

        print('QD pool size', len(gp.QD_pool))
        print('alleqsseen', len(local_alleqs))

        # save results and print
        saveme = printresults(train_targets, voc, calculus_mode, look_for, name)
        valrmse, bf = saveme.saveresults(newbin, replacements, i, gp.QD_pool, gp.bin_nscalar, local_alleqs, starttime, u, look_for)

        if valrmse <config.termination_nmrse:
            del results
            print('early stopping')
            return 'es', gp.QD_pool, local_alleqs, i, valrmse, bf
        if len(local_alleqs) > 10000000:
            del results
            print(' stopping bcs too many eqs seen')
            return 'stop', gp.QD_pool, local_alleqs, i, valrmse, bf

    return None, gp.QD_pool, local_alleqs, i, valrmse, bf


# -----------------------------------------------#
def main(params, train_targets, u, look_for, calculus_mode, maximal_size, name):

    # init target, dictionnaries, and meta parameters
    poolsize, delete_ar1_ratio, delete_ar2_ratio, extend_ratio, p_mutate, p_cross, addrandom, voc_no_a, voc_a, diff,\
    bin_length, bin_nscalar, bin_functions, bin_depth, bin_power, bin_target, bin_derivative, \
    bin_var, bin_norm, bin_cross, bin_dot_product = params


    if config.verifonegivenfunction:
        formula = 'A+A*x0'
        scalar_numbers, alla, rms = game_env.game_evaluate([1], formula, voc_a, train_targets, 'train', u, look_for)
        print('returns: ', scalar_numbers, alla, rms)
        time.sleep(4.1)

    prefix = str(int(10000000 * time.time()))

    if calculus_mode == 'scalar':

        # ------------------------ Step 1 : with integer scalars
        # init qd grid
        reinit_grid = False
        sttime = time.time()
        qdpoolname = 'QD_pool' + name + '_no_a_.txt'
        qdpool = init_grid(reinit_grid,  qdpoolname)

        gp = GP_QD(delete_ar1_ratio, delete_ar2_ratio, p_mutate, p_cross, poolsize, voc_no_a,
                   extend_ratio, bin_length, bin_nscalar, bin_functions, bin_depth, bin_power, bin_target,
                   bin_derivative, bin_var, bin_norm, bin_cross, bin_dot_product,
                   addrandom, calculus_mode, maximal_size, qdpool, None)

        iteration_no_a = config.iterationnoa
        stop, qdpool, alleqs_no_a, iter_no_a, valrmse, bf = exec(train_targets, u, voc_no_a,
                                                                 iteration_no_a, gp,
                                                                 look_for, calculus_mode, name, qdpoolname, sttime)

        # ------------------- step 2 -----------------------#
        #if target has not already been found, stop is None; then launch evoltion with free scalars A:
        if stop is None:
            qdpoolname_a = 'QD_pool' + name + '_a_.txt'

            # convert noA eqs into A eqs:
            if config.loadexistingmodel: #todo revoir le save qd/reload qd/en particulier a la transi
                QD_pool = init_grid(False, qdpoolname_a)
            else:
                initpool = convert_eqs(qdpool, voc_a, voc_no_a, diff, calculus_mode)

                # reinit gp class with 'A':
                gp = GP_QD(delete_ar1_ratio, delete_ar2_ratio, p_mutate, p_cross, poolsize, voc_a,
                   extend_ratio, bin_length, bin_nscalar, bin_functions, bin_depth, bin_power, bin_target,
                   bin_derivative, bin_var, bin_norm, bin_cross, bin_dot_product,
                   addrandom, calculus_mode, maximal_size, initpool, None)
                alleqs_change_mode, QD_pool, stop, valrmse, bf = eval_previous_eqs(train_targets, voc_a, u,
                                                                                   initpool, gp, look_for,
                                                                                   calculus_mode, name, qdpoolname_a, sttime)
    else:
        # mode is vectorial
        stop = None
        QD_pool = None

    if stop is None:
        qdpoolname_a = 'QD_pool' + name + '_a_.txt'
        #QD_pool = init_grid(False, qdpoolname_a)

        gp = GP_QD(delete_ar1_ratio, delete_ar2_ratio, p_mutate, p_cross, poolsize, voc_a,
               extend_ratio, bin_length, bin_nscalar, bin_functions, bin_depth, bin_power, bin_target,
               bin_derivative, bin_var, bin_norm, bin_cross, bin_dot_product,
               addrandom, calculus_mode, maximal_size, QD_pool, None)

        iteration_a = config.iterationa

        stop, qdpool, alleqs_a, iter_a, valrmse, bf =  exec(train_targets, u, voc_a,
                                                                 iteration_a, gp,
                                                                 look_for, calculus_mode, name, qdpoolname_a, sttime)