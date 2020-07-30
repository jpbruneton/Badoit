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
import utils_mainrun

# -----------------------------------------------#
if __name__ == '__main__':
    # don't display any output (CMA displays lots of warnings)
    noprint = False
    if noprint:
        utils_mainrun.kill_print()
    # ------------------

    pedagogical_sets = ['trivial_scalar_nodiff', 'trivial_scalar_diff', 'oh_scalar_diff', 'trivial_vectorial_diff',
                          'trivial_vectorial_no_diff', 'trivial_f_two_variables', 'koza2']
    actual_sets = ['hubblediagram', 'rindler_nodiff',
                   'rindler_diff', 'onebodyproblem','twobodyproblem', 'mond', 'testdim']
    set = actual_sets[0]

    target_first_paper = ['Keijzer1', 'Keijzer2', 'Keijzer14','Keijzer11', 'Keijzer15', 'Keijzer12', 'Keijzer4',
                          'Koza2', 'Koza3', 'Meier4', 'Nguyen2', 'Nguyen3', 'Nguyen4', 'Nguyen5', 'Nguyen6',
                          'Nguyen7', 'Nguyen9', 'Nonic', 'Pagie1', 'R1', 'R2', 'R3', 'Sine', 'Vladislavleva1']

    for set in target_first_paper:
        print('target is', set)
        filenames_train, explicit_time_dependence_allowed, use_first_derivatives, \
        use_function, look_for = utils_mainrun.define_set(set)

        # some expert knowledge doesnt make sense in non diff mode or scalar mode:
        if filenames_train[-1] == 'no_diff':
            explicit_time_dependence_allowed = True
            use_function = False
            use_first_derivatives = False

        expert_knowledge = [explicit_time_dependence_allowed, use_first_derivatives, use_function]

        # -------------------------------- init targets
        utils_mainrun.sanity_check(filenames_train) # check data format wrt options chosen
        n_coupled_targets = len(filenames_train) - 2
        calculus_mode = filenames_train[-2]

        n_variables, all_targets_name, train_targets, maxsize = utils_mainrun.init_targets(filenames_train)

        # solve :
        for u in range(n_coupled_targets):
            this_target = train_targets[u]
            maximal_size = maxsize[u] # the tree representing the equation in postfix has maximal_size nodes (leaves or internal)
            # main exec
            params = utils_mainrun.init_parameters(n_variables, all_targets_name, look_for, calculus_mode, maximal_size, u, expert_knowledge)
            run_one_target.main(params, train_targets, u, look_for, calculus_mode, maximal_size, set)