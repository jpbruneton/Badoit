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
def get_dic(n_targets, n_variables, all_targets_name, u, calculus_mode, look_for, expert_knowledge):
    explicit_time_dependence, use_first_derivatives = expert_knowledge

    # ============  arity 0 symbols =======================
    if calculus_mode == 'scalar':
        my_dic_scalar_number = ['A'] #A is going to be the generic name for one real scalar
        my_dic_vec_number = []
    else:
        my_dic_scalar_number = ['A']
        my_dic_vec_number = ['B']  #B is going to be the generic name for one real vector : this is really \vec{B}

    # allowing xpert knowledge :
    my_dic_special_scalar = []
    my_dic_variables = []
    if explicit_time_dependence:
        for i in range(n_variables):
            my_dic_variables.append('x'+str(i))
    else:
        pass

    # targets :
    my_dic_other_targets = all_targets_name[:u] + all_targets_name[u+1:]
    if look_for != 'find_function':
        my_dic_actual_target = [all_targets_name[u]]
    else:
        my_dic_actual_target = []

    #derivatives are considered as scalars here, not via a node operator (#later to do?)
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
    # basic functions
    my_dic_scalar_functions = config.fonctions

    if calculus_mode == 'vectorial':
        my_dic_vectorial_functions = ['la.norm('] # forms from E -> R
    else:
        my_dic_vectorial_functions = []

    # ============  arity 2 symbols =======================
    my_dic_scalar_operators = config.operators
    my_dic_power = ['**']

    if calculus_mode == 'vectorial':
        my_dic_vec_operators = ['np.vdot(', 'np.cross('] #operators from E*E -> R or E
    else:
        my_dic_vec_operators = []

    # ============ special algebraic symbols ======================= #occuring only via simplification
    my_dic_true_zero = ['zero']
    my_dic_neutral = ['neutral']
    my_dic_infinite = ['infinity']
    special_dic = my_dic_true_zero + my_dic_neutral + my_dic_infinite

    # --------------------------
    #concatenate the dics
    numbers_to_formula_dict = {'1' : 'halt'}
    bigdic =  my_dic_scalar_number + my_dic_vec_number + my_dic_special_scalar + my_dic_variables + my_dic_actual_target + my_dic_other_targets + my_dic_diff + my_dic_scalar_functions + my_dic_vectorial_functions+my_dic_scalar_operators + my_dic_vec_operators + my_dic_power
    print(bigdic)
    #arity 0:
    arity0dic = my_dic_scalar_number + my_dic_vec_number + my_dic_special_scalar + my_dic_variables + my_dic_actual_target\
                + my_dic_other_targets + my_dic_diff
    index0 = 2
    index1 = index0 + len(my_dic_scalar_number) + len(my_dic_vec_number)+len(my_dic_special_scalar)
    index2 = index1 + len(my_dic_variables)

    target_function_number = tuple([i for i in range(2 + len(my_dic_scalar_number) + len(my_dic_vec_number)+len(my_dic_special_scalar) + len(my_dic_variables), 2 + len(my_dic_scalar_number) + len(my_dic_vec_number)+len(my_dic_special_scalar) + len(my_dic_variables) + len(my_dic_actual_target)+len(my_dic_other_targets))])
    first_der_number = tuple([i for i in range(2 + len(my_dic_scalar_number) + len(my_dic_vec_number)+len(my_dic_special_scalar) + len(my_dic_variables) + len(my_dic_actual_target)+len(my_dic_other_targets), 2 + len(my_dic_scalar_number) + len(my_dic_vec_number)+len(my_dic_special_scalar) + len(my_dic_variables) + len(my_dic_actual_target)+len(my_dic_other_targets) + len(my_dic_diff))])

    if calculus_mode == 'vectorial' and len(my_dic_vec_number) !=0:
        vectorial_numbers = [index0 +  len(my_dic_scalar_number)]
    if calculus_mode == 'vectorial':
        vectorial_numbers.extend([i for i in range(index2, 2 + len(arity0dic))])
        arity0_vec = tuple(deepcopy(vectorial_numbers))

    if explicit_time_dependence:
        arity0_novec = (index0, index0+1+len(my_dic_vec_number)+len(my_dic_special_scalar))
    else:
        arity0_novec = [index0] #oubli de la fonction!!

    pure_numbers = tuple([i for i in range(index0, index1)])
    var_numbers = tuple([i for i in range(index1, index2)])

    #arity 1 and 2 :
    arity1dic = my_dic_scalar_functions + my_dic_vectorial_functions
    arity2dic = my_dic_scalar_operators + my_dic_vec_operators + my_dic_power
    a0, a1, a2  = len(arity0dic), len(arity1dic), len(arity2dic)
    arity_1_novec = [i for i in range(2 + a0, 2 + a0 + len(my_dic_scalar_functions))]

    # dont change order of operations!
    plusnumber = 2 + a0 + a1
    minusnumber = 3 + a0 + a1
    multnumber = 4 + a0 + a1
    divnumber = 5 + a0 + a1

    if calculus_mode == 'vectorial':
        vectorial_numbers.extend([i for i in range(2+a0+len(my_dic_scalar_functions), 2+a0+a1)])
        arity1_vec = 2+a0+len(my_dic_scalar_functions)
        vectorial_numbers.extend([i for i in range(2+a0+a1+len(my_dic_scalar_operators), 2+a0+a1+len(my_dic_scalar_operators) + len(my_dic_vec_operators))])
        arity2_vec = tuple([i for i in range(2+a0+a1+len(my_dic_scalar_operators), 2+a0+a1+len(my_dic_scalar_operators) + len(my_dic_vec_operators))] + [plusnumber, minusnumber])

    if calculus_mode == 'vectorial':
        norm_number = 2 + a0 + a1 - 1
        vectorial_numbers = tuple(vectorial_numbers)
        print(vectorial_numbers)
    else:
        vectorial_numbers = None
        arity0_vec = None
        arity1_vec = None
        arity2_vec = None
        norm_number = None

    arity0symbols = tuple([i for i in range(2, 2 + a0)])
    arity1symbols = tuple([i for i in range(2 + a0, 2 + a0 + a1)])
    arity2symbols = tuple([i for i in range(2 + a0 + a1, 2 + a0 + a1 + a2)])
    arity2symbols_no_power = tuple([i for i in range(2 + a0 + a1, 2 + a0 + a1 + a2 -1)])


    if calculus_mode == 'vectorial':
        dotnumber = 6+a0+a1
        wedgenumber = 7+a0+a1
    else:
        dotnumber = None
        wedgenumber = None

    if calculus_mode == 'vectorial':
        arity2symbols_novec = tuple([x for x in arity2symbols if x not in arity2_vec]+ [plusnumber, minusnumber])
    else:
        arity2symbols_novec = arity2symbols

    #dont change the previous order or change this accordingly
    power_number = 1 + a0 + a1 + a2
    true_zero_number = 2 + a0 + a1 + a2
    neutral_element = 3 + a0 + a1 + a2
    infinite_number = 4 + a0 + a1 + a2

    # finally
    all_my_dics = arity0dic + arity1dic + arity2dic + special_dic
    for i in range(len(all_my_dics)):
        numbers_to_formula_dict.update({str(i+2): all_my_dics[i]})

    terminalsymbol = 1
    print('ee', arity2_vec)
    all = [arity0symbols, arity1symbols, arity2symbols, true_zero_number, neutral_element, \
           infinite_number, terminalsymbol, pure_numbers, arity2symbols_no_power, power_number, var_numbers, \
           plusnumber, minusnumber, multnumber, divnumber, norm_number, dotnumber, wedgenumber, vectorial_numbers, \
           arity0_vec, arity0_novec, arity1_vec, arity2_vec, arity2symbols_novec, arity_1_novec, target_function_number, first_der_number]


    print('toi dic')
    for el in all:
        print(el)
    return numbers_to_formula_dict, arity0symbols, arity1symbols, arity2symbols, true_zero_number, neutral_element, \
           infinite_number, terminalsymbol, pure_numbers, arity2symbols_no_power, power_number, var_numbers, \
           plusnumber, minusnumber, multnumber, divnumber, norm_number, dotnumber, wedgenumber, vectorial_numbers, \
           arity0_vec, arity0_novec, arity1_vec, arity2_vec, arity2symbols_novec, arity_1_novec, target_function_number, first_der_number

# example this might return (two vec targets F0, F1, one scalar variable 'time' (x0), first derivatives, etc
#{'1': 'halt', '2': 'A', '3': 'B', '4': 'x0', '5': 'F0', '
# 6': 'np.sqrt(', '7': 'np.exp(', '8': 'np.log(', '9': 'np.arctan(', '10': 'np.tanh(',
# '11': 'la.norm(',
# '12': '+', '13': '-', '14': '*', '15': '/', '16': 'np.vdot(', '17': 'np.cross(',
# '18': '**', '19': 'zero', '20': 'neutral', '21': 'infinity'}
