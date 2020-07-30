#  ======================== CMA-Based Symbolic Regressor ========================== #
# Project:          Symbolic regression for physics
# Name:             AST.py
# Authors:          Jean-Philippe Bruneton
# Date:             2020
# License:          BSD 3-Clause License
# ============================================================================ #


# ================================= PREAMBLE ================================= #
# Packages
import run_one_target
import config
from Targets import Target, Voc
import numpy as np
import pickle

# -----------------------------------------------#
def init_parameters(n_variables, all_targets_name, look_for, calculus_mode, maximal_size, u, expert_knowledge):

    # init dictionnaries
    voc_no_a = Voc(u, n_variables, all_targets_name, calculus_mode, maximal_size, look_for, expert_knowledge, 'no_A')
    voc_a = Voc(u, n_variables, all_targets_name, calculus_mode, maximal_size, look_for, expert_knowledge, 'A')
    print('we work with voc: ', voc_no_a.numbers_to_formula_dict)
    print('and then with voc: ', voc_a.numbers_to_formula_dict)

    diff = len(voc_no_a.numbers_to_formula_dict) - len(voc_a.numbers_to_formula_dict)

    # and metaparameters
    poolsize = config.qd_init_pool_size
    if config.MAX_DEPTH ==1:
        delete_ar1_ratio = 0.2
    elif config.MAX_DEPTH == 2:
        delete_ar1_ratio = 0.4
    else:
        delete_ar1_ratio = 0.8 # increasing values randomly guessed in order to prevent too many nested functions

    delete_ar2_ratio = 0.1
    extend_ratio = config.extendpoolfactor
    p_mutate = 0.4
    p_cross = 0.8

    bin_length = maximal_size # number of bins for length of an eq
    bin_nscalar = maximal_size  # number of bins for number of free scalars
    bin_functions = 10 # number of bins for number of fonctions
    if config.smallgrid:
        extra_grid = 0
    else:
        extra_grid = 2
    bin_power = extra_grid  # number of bins for number of powers
    bin_target = extra_grid
    bin_depth = config.MAX_DEPTH
    bin_var = 20
    bin_derivative = extra_grid
    addrandom = config.add_random
    bin_norm, bin_cross, bin_dot_product = [extra_grid]*3

    params = [poolsize, delete_ar1_ratio, delete_ar2_ratio, extend_ratio, p_mutate, p_cross, addrandom, voc_no_a, voc_a, diff,
              bin_length, bin_nscalar, bin_functions, bin_depth, bin_power, bin_target, bin_derivative, bin_var,
              bin_norm, bin_cross, bin_dot_product]
    return params

# -----------------------------------------------#
def kill_print():
    import sys
    class writer(object):
        log = []

        def write(self, data):
            self.log.append(data)

    logger = writer()
    sys.stdout = logger
    sys.stderr = logger

# -----------------------------------
def sanity_check(filenames_train):

    # todo later : this does not allow mixed files, like solve \vec{x}(t) as a diff eq that may also depend
    # todo from an auxiliary scalar variable q(t) in another file

    training_nvar = []
    for u in range(len(filenames_train) - 2):
        file = open(filenames_train[u], 'rb')
        dat = pickle.load(file)
        training_nvar.append(dat['n_variables'])

    if filenames_train[-1] == 'diff':
        if filenames_train[-2] == 'vectorial':
            for nvar in training_nvar:
                if nvar !=1:
                    print('training files should all have the format : time, x(t), y(t), z(t)')
                    raise ValueError
        if filenames_train[-2] == 'scalar':
            for nvar in training_nvar:
                if nvar !=1:
                    print('training files should all have the format : time, f(t)')
                    raise ValueError

    elif filenames_train[-1] == 'no_diff':
        if filenames_train[-2] == 'vectorial':
            for nvar in training_nvar:
                if nvar != 1:
                    print('training files should all have the format : time, x(t), y(t), z(t)')
                    raise ValueError
        if filenames_train[-2] == 'scalar':
            for nvar in training_nvar:
                if nvar > 3:
                    print('fitting of a function, f(a,b,c,..) not supported with more than three variables')
                    raise ValueError

    else:
        print('mode shd be diff or no_diff check for typos')
        raise ValueError

# ----------------------------------------
def define_set(set):
    possible_modes = ['find_function', 'find_1st_order_diff_eq', 'find_2nd_order_diff_eq']
    maxsize = []
    # warning vectorial/scalar; diff/nodiff must be specified. Vectorial are assumed 3D (only)
    # example of finding a non_diff eq for one target/one variable (thus in scalar mode) : affine law
    # ============================ trivial pedagogical sets =======================================
    if set in ['Keijzer1', 'Keijzer2', 'Keijzer5','Keijzer14','Keijzer11', 'Keijzer15', 'Keijzer12', 'Keijzer4',
                          'Koza2', 'Koza3', 'Meier4', 'Nguyen2', 'Nguyen3', 'Nguyen4', 'Nguyen5', 'Nguyen6',
                          'Nguyen7', 'Nguyen9', 'Nonic', 'Pagie1', 'R1', 'R2', 'R3', 'Sine', 'Vladislavleva1', 'Vladislavleva3']:

        filenames_train = ['data_loader/'+ set +'.txt'] + ['scalar', 'no_diff']
        explicit_time_dependence_allowed = True
        use_first_derivatives = False
        use_fonction = False
        look_for = possible_modes[0]

    elif set == 'trivial_scalar_nodiff':
        filenames_train = ['data_loader/simple_affine.txt', 'scalar', 'no_diff']
        explicit_time_dependence_allowed = True
        use_first_derivatives = False
        use_fonction = False
        look_for = possible_modes[0]

    # example of finding a diff eq for one target/one variable (thus in scalar mode) : \ddot{f(t)} = affine law
    elif set == 'trivial_scalar_diff':
        filenames_train = ['data_loader/simple_poly3.txt', 'scalar', 'diff']
        explicit_time_dependence_allowed = True
        use_first_derivatives = False # helps but not required
        use_fonction = False
        look_for = possible_modes[2]

    elif set == 'oh_scalar_diff':
        filenames_train = ['data_loader/simple_oscillator.txt', 'scalar', 'diff']
        explicit_time_dependence_allowed = True
        use_fonction = True
        use_first_derivatives = True # helps but not required
        look_for = possible_modes[2]

    # example of finding a diff eq for one target in vectorial mode : \vec{\ddot{x}(t)} = afine law = some_vec + some_vec * t
    elif set == 'trivial_vectorial_diff':
        filenames_train = ['data_loader/dummy_diff_vec.txt', 'vectorial', 'diff']
        explicit_time_dependence_allowed = True
        use_fonction = False
        use_first_derivatives = False  # helps but not required
        look_for = possible_modes[2]

    # example of finding a diff eq for one target in vectorial mode : target here is : \vec{x(t)} = afine law = some_vec + some_vec * t
    elif set == 'trivial_vectorial_no_diff':
        filenames_train = ['data_loader/dummy_nodiff_vec.txt', 'vectorial',
                           'no_diff']
        explicit_time_dependence_allowed = True
        use_fonction = False
        use_first_derivatives = False
        look_for = possible_modes[0]

    # example of finding f(x,y) = .. : here is A*x + B*y
    elif set == 'trivial_f_two_variables':
        filenames_train = ['data_loader/f_two_variables.txt', 'scalar', 'no_diff']
        explicit_time_dependence_allowed = True
        use_fonction = False
        use_first_derivatives = True
        look_for = possible_modes[0]

    elif set == 'koza2':
        filenames_train = ['data_loader/Koza2.txt', 'scalar', 'no_diff']
        explicit_time_dependence_allowed = True
        use_fonction = False
        use_first_derivatives = False
        look_for = possible_modes[0]
    # ============================ actual non trivial sets =======================================

    # example of finding a non trivial diff eq for one target in vectorial mode
    elif set == 'onebodyproblem':
        filenames_train = ['data_loader/kepler_1_body.txt', 'vectorial', 'diff']
        explicit_time_dependence_allowed = True  # this is expert knowledge since the target is \vec{\ddot{F}(t)} = k \vec{x}/\norm{x}^3 with no time dependence
        use_first_derivatives = True  # idem : no dot(f) in the actual solution (this helps)
        use_fonction = True
        look_for = possible_modes[2]

    elif set == 'rindler_nodiff':
        filenames_train = ['data_loader/Rindler_30000.txt', 'scalar', 'no_diff']
        explicit_time_dependence_allowed = True  # this is expert knowledge since the target is \vec{\ddot{F}(t)} = k \vec{x}/\norm{x}^3 with no time dependence
        use_first_derivatives = False  # idem : no dot(f) in the actual solution (this helps)
        use_fonction = False
        look_for = possible_modes[0]

    elif set == 'rindler_diff':
        filenames_train = ['data_loader/Rindler_100000.txt', 'scalar', 'diff']
        explicit_time_dependence_allowed = True  # this is expert knowledge since the target is \vec{\ddot{F}(t)} = k \vec{x}/\norm{x}^3 with no time dependence
        use_first_derivatives = True  # idem : no dot(f) in the actual solution (this helps)
        use_fonction = False
        look_for = possible_modes[2]

    # example of finding a coupled diff equations for two targets
    elif set == 'twobodyproblem':
        filenames_train = ['data_loader/Newtonian2body_body1.csv', 'data_loader/Newtonian2body_body2.csv', 'vectorial', 'diff']
        explicit_time_dependence_allowed = True
        use_first_derivatives = True
        use_fonction = False
        look_for = possible_modes[2]

    # example mono target no_diff with dimensional analysis
    elif set == 'mond':
        filenames_train = ['data_loader/mond_bin.csv', 'scalar', 'no_diff']
        explicit_time_dependence_allowed = False
        use_first_derivatives = False
        use_fonction = False
        look_for = possible_modes[0]

    elif set == 'hubblediagram':
        filenames_train = ['data_loader/hubblediagram.txt', 'scalar', 'no_diff']
        explicit_time_dependence_allowed = True
        use_first_derivatives = False
        use_fonction = False
        look_for = possible_modes[0]

    else:
        print('set of data not recognized')
        raise ValueError

    return filenames_train, explicit_time_dependence_allowed, use_first_derivatives, use_fonction, look_for
# ----------------------------------------------
def init_targets(filenames_train):
    '''
    for coupled systems, eg. file0 is t, x0, y0, z0 and file1 is t, x1, y1, z1
    defines 2 target names F0 == vec(x,y,z)_0 and F1 idem in vec mode
    and their corresponding data of the form [F0, fata for F0, data for FO_dot, data for F_0_ddot]
    etc
    we could also solve in scalar mode a file  t, x, y, z instaed of vectorially, but then its equivalent to have
    3 different files t,x ; t, y ; t, z
    '''
    definetarget = Target(filenames_train)
    train = definetarget.targets
    n_variables = definetarget.n_variables
    alltraintargets = []
    calc_mode = filenames_train[-2]
    maxsize = definetarget.maxsize

    for u in range(len(filenames_train) - 2):
        if calc_mode == 'vectorial':
            name = 'F'+str(u) # capital F represent the vectorial target
        else:
            name = 'f'+str(u) # and a scalar resp.
        onetarget =  [name, train[u][0], train[u][1],train[u][2],train[u][3], train[u][4]] # name, variable, fonction, first der, second der, ranges
        alltraintargets.append(onetarget)

    all_targets_name = [alltraintargets[u][0] for u in range(len(alltraintargets))]
    return n_variables, all_targets_name, alltraintargets, maxsize
