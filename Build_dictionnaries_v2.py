#  ======================== CMA-Based Symbolic Regressor ========================== #
# Project:          Symbolic regression for physics
# Name:             AST.py
# Authors:          Jean-Philippe Bruneton
# Date:             2020
# License:          BSD 3-Clause License
# ============================================================================ #


# ================================= PREAMBLE ================================= #
# Packages
import config
from copy import deepcopy


# returns numbers corresponding to characters used and as specified in config
def get_dic(modescalar, n_targets, n_variables, all_targets_name, u, calculus_mode, look_for, expert_knowledge):
    explicit_time_dependence, use_first_derivatives, use_function = expert_knowledge




    # ============  arity 0 symbols =======================
    if calculus_mode == 'scalar':
        if modescalar == 'no_A':
            my_dic_scalar_number = config.list_scalars
        else:
            my_dic_scalar_number = ['A'] #A is going to be the generic name for one real scalar
        my_dic_vec_number = []
    else:
        # no_A not allowed yet in vec mode #todo
        my_dic_scalar_number = ['A']
        my_dic_vec_number = ['B']  #B is going to be the generic name for one real vector : this is really \vec{B}

    # allowing expert knowledge :
    my_dic_special_scalar = []
    my_dic_variables = []
    if explicit_time_dependence:
        for i in range(n_variables):
            my_dic_variables.append('x'+str(i))

    # targets :

    if use_function:
        my_dic_other_targets = all_targets_name[:u] + all_targets_name[u+1:]
    else:
        my_dic_other_targets = []
    if look_for != 'find_function':
        if use_function:
            my_dic_actual_target = [all_targets_name[u]]
        else:
            my_dic_actual_target = []
    else:
        my_dic_actual_target = []

    # derivatives are considered as scalars here, not via a node operator (#later to do?)
    if look_for == 'find_1st_order_diff_eq':
        maxder = 1
    elif look_for == 'find_2nd_order_diff_eq':
        maxder = 2
    else:
        maxder = 0

    my_dic_diff = []
    ders = [['']]
    while len(ders) < maxder:
        loc_der = []
        pre_der = ders[-1]
        for elem in pre_der:
                loc_der.append('d' + elem + '_x0')
        ders.append(loc_der)

    for u in range(n_targets):
        for i in range(1, len(ders)):
            for der in ders[i]:
                if calculus_mode == 'scalar':
                    my_dic_diff.append(der + '_f' + str(u)) #scalar targets are with a lowercase f, vectorial targets with F
                else:
                    my_dic_diff.append(der + '_F' + str(u))

    if not use_first_derivatives:
        my_dic_diff = []

    # ============  arity 1 symbols =======================
    # usual functions
    my_dic_scalar_functions = config.fonctions
    my_dic_vectorial_functions = []
    if calculus_mode == 'vectorial':
        my_dic_vectorial_functions = ['la.norm('] # forms from E -> R

    # ============  arity 2 symbols =======================
    my_dic_scalar_operators = config.operators
    my_dic_power = ['**']
    my_dic_vec_operators = []
    if calculus_mode == 'vectorial':
        my_dic_vec_operators = ['np.vdot(', 'np.cross('] #operators from E*E -> R or E
    if calculus_mode == 'vectorial':
        my_dic_vec_operators_extended = my_dic_vec_operators + ['+', '-']  # operators from E*E -> R or E

    # ============ special algebraic symbols ======================= #occuring only via simplification
    my_dic_true_zero = ['zero']
    my_dic_neutral = ['neutral']
    my_dic_infinite = ['infinity']
    special_dic = my_dic_true_zero + my_dic_neutral + my_dic_infinite

    # --------------------------
    #concatenate the dics
    numbers_to_formula_dict = {'1' : 'halt'}
    symbol_to_number = {}
    all_symbols = my_dic_scalar_number + my_dic_vec_number + my_dic_special_scalar \
                  + my_dic_variables + my_dic_actual_target + my_dic_other_targets \
                  + my_dic_diff + my_dic_scalar_functions + my_dic_vectorial_functions\
                  + my_dic_scalar_operators + my_dic_vec_operators + my_dic_power+special_dic

    c=2
    for elem in all_symbols:
        numbers_to_formula_dict.update({str(c) : elem })
        symbol_to_number.update({elem : c})
        c += 1

    # get sub_dictionnaries:

    #arity 0:
    arity0dic = my_dic_scalar_number + my_dic_vec_number + my_dic_special_scalar + my_dic_variables + my_dic_actual_target\
                + my_dic_other_targets + my_dic_diff
    arity0symbols = []
    for elem in arity0dic:
        arity0symbols.append(symbol_to_number[elem])
    arity0symbols = tuple(arity0symbols)

    target_function_number = []
    for elem in my_dic_actual_target+my_dic_other_targets:
        target_function_number.append(symbol_to_number[elem])
    target_function_number = tuple(target_function_number)

    first_der_number = []
    for elem in my_dic_diff:
        first_der_number.append(symbol_to_number[elem])
    first_der_number = tuple(first_der_number)

    arity0_vec = []
    for elem in my_dic_vec_number:
        arity0_vec.append(symbol_to_number[elem])
    if calculus_mode == 'vectorial': #then targets are also vectors
        arity0_vec.extend(target_function_number)
        arity0_vec.extend(first_der_number)
    arity0_vec = tuple(arity0_vec)

    arity0_novec = tuple([x for x in arity0symbols if x not in arity0_vec])

    pure_numbers = []
    for elem in my_dic_scalar_number + my_dic_vec_number + my_dic_special_scalar:
        pure_numbers.append(symbol_to_number[elem])
    pure_numbers = tuple(pure_numbers)

    var_numbers = []
    for elem in my_dic_variables:
        var_numbers.append(symbol_to_number[elem])
    var_numbers = tuple(var_numbers)

    #arity 1 and 2 :
    arity1symbols = []
    for elem in my_dic_scalar_functions + my_dic_vectorial_functions:
        arity1symbols.append(symbol_to_number[elem])
    arity1symbols = tuple(arity1symbols)

    arity_1_novec = []
    for elem in my_dic_scalar_functions:
        arity_1_novec.append(symbol_to_number[elem])
    arity_1_novec = tuple(arity_1_novec)

    arity_1_vec = []
    for elem in my_dic_vectorial_functions:
        arity_1_vec.append(symbol_to_number[elem])
    arity_1_vec = tuple(arity_1_vec)

    arity2symbols = []
    for elem in my_dic_scalar_operators + my_dic_vec_operators + my_dic_power:
        arity2symbols.append(symbol_to_number[elem])
    arity2symbols = tuple(arity2symbols)

    arity2symbols_nopower= []
    for elem in my_dic_scalar_operators + my_dic_vec_operators :
        arity2symbols_nopower.append(symbol_to_number[elem])
    arity2symbols_nopower = tuple(arity2symbols_nopower)

    arity_2_novec = []
    for elem in my_dic_scalar_operators+ my_dic_power:
        arity_2_novec.append(symbol_to_number[elem])
    arity_2_novec = tuple(arity_2_novec)

    arity_2_vec = []
    if calculus_mode == 'vectorial':
        for elem in my_dic_vec_operators_extended:
            arity_2_vec.append(symbol_to_number[elem])
    arity_2_vec = tuple(arity_2_vec)
    # plus and minus also counts, here, so first :

    plusnumber = tuple([symbol_to_number['+']])
    minusnumber = tuple([symbol_to_number['-']])
    multnumber = tuple([symbol_to_number['*']])
    divnumber = tuple([symbol_to_number['/']])
    powernumber = tuple([symbol_to_number['**']])

    if calculus_mode == 'vectorial':
        dotnumber = tuple([symbol_to_number['np.vdot(']])
        wedgenumber = tuple([symbol_to_number['np.cross(']])
        norm_number = tuple([symbol_to_number['la.norm(']])

    else:
        dotnumber = tuple([None])
        wedgenumber = tuple([None])
        norm_number = tuple([None])

    #dont change the previous order or change this accordingly
    true_zero_number = tuple([symbol_to_number['zero']])
    neutral_element = tuple([symbol_to_number['neutral']])
    infinite_number = tuple([symbol_to_number['infinity']])

    terminalsymbol = tuple([1])


    all = [arity0symbols, arity1symbols, arity2symbols, true_zero_number, neutral_element, \
           infinite_number, terminalsymbol, pure_numbers, arity2symbols_nopower, powernumber, var_numbers, \
           plusnumber, minusnumber, multnumber, divnumber, norm_number, dotnumber, wedgenumber, arity0_vec, \
           arity0_vec, arity0_novec, arity_1_vec, arity_2_vec, arity_2_novec, arity_1_novec, target_function_number, first_der_number]


    #for el in all:
    #    print(el)

    return [numbers_to_formula_dict] + all

# example this might return (two vec targets F0, F1, one scalar variable 'time' (x0), first derivatives, etc
#{'1': 'halt', '2': 'A', '3': 'B', '4': 'x0', '5': 'F0', '
# 6': 'np.sqrt(', '7': 'np.exp(', '8': 'np.log(', '9': 'np.arctan(', '10': 'np.tanh(',
# '11': 'la.norm(',
# '12': '+', '13': '-', '14': '*', '15': '/', '16': 'np.vdot(', '17': 'np.cross(',
# '18': '**', '19': 'zero', '20': 'neutral', '21': 'infinity'}
