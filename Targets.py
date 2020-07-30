#  ======================== CMA-Based Symbolic Regressor ========================== #
# Project:          Symbolic regression for physics
# Name:             AST.py
# Authors:          Jean-Philippe Bruneton
# Date:             2020
# License:          BSD 3-Clause License
# ============================================================================ #


# ================================= PREAMBLE ================================= #
# Packages
import numpy as np
import Build_dictionnaries_v2 as Build_dictionnaries
#import Build_dictionnaries
import Simplification_rules
import pickle
import config
from scipy import interpolate

# ============================================================================ #

class Target():

    def __init__(self, filenames):
        self.filenames = filenames
        self.targets = []
        self.diffmode = filenames[-1]
        self.calculus_mode = filenames[-2]
        self.maxsize = []

        for u in range(len(filenames) - 2):
            if self.diffmode == 'no_diff':
                self.targets.append(self._define_nondiff_targetfromfile(u))
            else:
                self.targets.append(self._define_diff_targetfromfile(u))

    # -----------------------------------------
    def _define_nondiff_targetfromfile(self, u):
        '''
        Relevant for finding a symbolic eq for the data, not using any diff operator
        This assumes the file having the form (in scalar mode) : first p columns : the variables (max supported : 3), then one target (more not supported yet)
        eg. x, y, z, f(x,y,z) ; if vectorial, it must be t, x, y, z; objective function is \vec{x}(t) = ...
        :return: the list of [variable, the vector of targets (\vec x, vec y \vec z), the vector of its first derivatives, and the vector of its second derivatives]
        '''
        file = open(self.filenames[u], 'rb')
        my_dict = pickle.load(file)
        self.maxsize.append(my_dict['maxlen'])
        if self.calculus_mode == 'vectorial':
            # file must be :  variable (named t) , x, y, z
            self.n_variables = 1
            x0 = my_dict['x0']
            rangex = np.max(x0) - np.min(x0)
            x0 = x0 / rangex

            f0 = my_dict['f0']
            rangef0 = np.max(f0) - np.min(f0)
            f0 = f0 / rangef0

            f1 = my_dict['f1']
            rangef1 = np.max(f1) - np.min(f1)
            f1 = f1 / rangef1

            f2 = my_dict['f2']
            rangef2 = np.max(f2) - np.min(f2)
            f2 = f2 / rangef2
            return [[x0], np.transpose(np.array([f0, f1, f2])), None, None, [rangex, rangef0, rangef1, rangef2]]

        else:
            self.n_variables = my_dict['n_variables']
            # ------------------------------------------------#
            if self.n_variables == 1:
                x = my_dict['x0']
                f0 = my_dict['f0']
                #print(x)
                #print(f0)
                rangef = np.max(f0) - np.min(f0)
                rangex = np.max(x) - np.min(x)
                f0 = f0/rangef
                x = x/rangex
                tck = interpolate.splrep(x, f0, s=0)
                # even if in a non diff mode, the comparison between target and first derivatives can enter the cost to speed up convergence

                f_first_der = interpolate.splev(x, tck, der=1)
                f_sec_der = interpolate.splev(x, tck, der=2)
                return [[x], np.transpose(np.array([f0])), np.transpose(np.array([f_first_der])), np.transpose(np.array([f_sec_der])), [rangex, rangef]]

            # ------------------------------------------------#
            # for more than 1 variable we dont provide the gradients
            elif self.n_variables == 2:
                X = my_dict['x0']
                Y = my_dict['x1'] # these are meshgrids
                rangex = np.max(X) - np.min(X)
                rangey = np.max(Y) - np.min(Y)
                f0 = my_dict['f0']
                rangef = np.max(f0) - np.min(f0)

                return [[X/rangex, Y/rangey], f0/rangef, None, None, [rangex, rangey, rangef]]

            # ------------------------------------------------#
            elif self.n_variables == 3:
                X = my_dict['x0']
                Y = my_dict['x1']  # these are meshgrids
                Z = my_dict['x2']  # these are meshgrids
                f0 = my_dict['f0']
                rangex = np.max(X) - np.min(X)
                rangey = np.max(Y) - np.min(Y)
                rangez = np.max(Z) - np.min(Z)
                f0 = my_dict['f0']
                rangef = np.max(f0) - np.min(f0)
                return [[X/rangex, Y/rangey, Z/rangez], f0/rangef, None, None, [rangex, rangey, rangez, rangef]]

    # ----------------------------------------------
    def _define_diff_targetfromfile(self, u):
        '''
        Relevant for finding a diff eq on the data (of one variable only, more is not supported)
        This assumes the file having the form : column 0 : the variable, and then 1,2, or 3 columns for the target functions : eg. x(t), y(t), z(t)
        works both for scalar or vectorial mode
        :return: the list of [variable, the vector of targets (\vec x, vec y \vec z), the vector of its first derivatives, and the vector of its second derivatives]
        '''

        file = open(self.filenames[u], 'rb')
        my_dict = pickle.load(file)
        self.maxsize.append(my_dict['maxlen'])
        self.n_variables = 1 #diff are always mono var
        t = my_dict['x0']
        ranget = np.max(t) - np.min(t)
        f0 = my_dict['f0']
        rangef0 = np.max(f0) - np.min(f0)
        target_functions = [f0]
        t = t/ranget
        f0 = f0/rangef0
        if False:
            tck = interpolate.splrep(t, f0, s=0)
            new_x = np.linspace(t[0], t[-1], num=20000)  # todo pourquoi entre pemier x et 1?? mond spec i guess
            new_f = interpolate.splev(new_x, tck)
            tck = interpolate.splrep(new_x, new_f, s=0)
            f_first_der = interpolate.splev(new_x, tck, der=1)
            f_sec_der = interpolate.splev(new_x, tck, der=2)
            return [[new_x], np.transpose(np.array([new_f])), np.transpose(np.array([f_first_der])),
                    np.transpose(np.array([f_sec_der])),[ranget, rangef0]]

        # derivatives are computed by using an interpolation of the target #these are derivatives of the scaled functions, it needs appropriate descaling later
        tck = interpolate.splrep(t, f0, s=0)
        f0_first_der = interpolate.splev(t, tck, der=1)
        f0_second_der = interpolate.splev(t, tck, der=2)
        first_derivatives = [f0_first_der]
        second_derivatives = [f0_second_der]

        if self.calculus_mode == 'vectorial':
            f1 = my_dict['f1']
            rangef1 = np.max(f1) - np.min(f1)
            f1 = f1/rangef1
            target_functions.append(f1)
            tck = interpolate.splrep(t, f1, s=0)
            f1_first_der = interpolate.splev(t, tck, der=1)
            f1_sec_der = interpolate.splev(t, tck, der=2)
            first_derivatives.append(f1_first_der)
            second_derivatives.append(f1_sec_der)

            f2 = my_dict['f2']
            rangef2 = np.max(f2) - np.min(f2)
            f2 = f2/rangef2
            target_functions.append(f2)
            tck = interpolate.splrep(t, f2, s=0)
            f2_first_der = interpolate.splev(t, tck, der=1)
            f2_sec_der = interpolate.splev(t, tck, der=2)
            first_derivatives.append(f2_first_der)
            second_derivatives.append(f2_sec_der)

            return [[t], np.transpose(np.array(target_functions)), np.transpose(np.array(first_derivatives)),
                    np.transpose(np.array(second_derivatives)), [ranget, rangef0, rangef1, rangef2]]
        else:
            return [[t], np.transpose(np.array(target_functions)), np.transpose(np.array(first_derivatives)),
                    np.transpose(np.array(second_derivatives)), [ranget, rangef0]]

# ============================================================================ #
# ============================================================================ #

class Voc():
    def __init__(self, u, n_variables, all_targets_name, calculus_mode,
                 maximal_size, look_for, expert_knowledge, modescalar):
        self.calculus_mode = calculus_mode
        self.maximal_size = maximal_size
        self.all_targets_name = all_targets_name
        self.n_targets = len(all_targets_name)
        self.look_for = look_for
        self.expert_knowledge = expert_knowledge
        self.n_variables = n_variables
        self.modescalar = modescalar # 'A', 'no_A'

        self.numbers_to_formula_dict, self.arity0symbols, self.arity1symbols, self.arity2symbols, self.true_zero_number, self.neutral_element, \
        self.infinite_number, self.terminalsymbol, self.pure_numbers, self.arity2symbols_no_power, self.power_number, self.var_numbers, \
        self.plusnumber, self.minusnumber, self.multnumber, self.divnumber, self.norm_number, self.dot_number, self.wedge_number, \
        self.vectorial_numbers, self.arity0_vec, self.arity0_novec, self.arity1_vec, self.arity2_vec, self.arity2novec, self.arity1_novec,\
            self.targetfunction_number, self.first_der_number\
            = Build_dictionnaries.get_dic(self.modescalar, self.n_targets, self.n_variables,
                                          self.all_targets_name, u, self.calculus_mode, self.look_for, self.expert_knowledge)


        #todo redo later
        #self.mysimplificationrules, self.maxrulesize = self.create_dic_of_simplifs()

    def replacemotor(self, toreplace,replaceby, k):
        firstlist = []
        secondlist = []
        for elem in toreplace:
            if elem == 'zero':
                firstlist.append(self.true_zero_number)
            elif elem == 'neutral':
                firstlist.append(self.neutral_element)
            elif elem == 'infinite':
                firstlist.append(self.infinite_number)
            elif elem == 'scalar':
                firstlist.append(self.pure_numbers[0])
            elif elem == 'mult':
                firstlist.append(self.multnumber)
            elif elem == 'plus':
                firstlist.append(self.plusnumber)
            elif elem == 'minus':
                firstlist.append(self.minusnumber)
            elif elem == 'div':
                firstlist.append(self.divnumber)
            elif elem == 'variable':
                firstlist.append(self.var_numbers[k])
            elif elem == 'arity0':
                firstlist.append(self.arity0symbols[k])
            elif elem == 'fonction':
                firstlist.append(self.arity1symbols[k])
            elif elem == 'allops':
                firstlist.append(self.arity2symbols[k])
            elif elem == 'power':
                firstlist.append(self.power_number)
            elif elem == 'one':
                firstlist.append(self.pure_numbers[0])
            elif elem == 'two':
                firstlist.append(self.pure_numbers[1])
            else:
                print('bug1', elem)

        for elem in replaceby:
            if elem == 'zero':
                secondlist.append(self.true_zero_number)
            elif elem == 'neutral':
                secondlist.append(self.neutral_element)
            elif elem == 'infinite':
                secondlist.append(self.infinite_number)
            elif elem == 'scalar':
                secondlist.append(self.pure_numbers[0])
            elif elem == 'mult':
                secondlist.append(self.multnumber)
            elif elem == 'plus':
                secondlist.append(self.plusnumber)
            elif elem == 'minus':
                secondlist.append(self.minusnumber)
            elif elem == 'div':
                secondlist.append(self.divnumber)
            elif elem == 'variable':
                secondlist.append(self.var_numbers[k])
            elif elem == 'arity0':
                secondlist.append(self.arity0symbols[k])
            elif elem == 'fonction':
                secondlist.append(self.arity1symbols[k])
            elif elem == 'allops':
                secondlist.append(self.arity2symbols[k])
            elif elem == 'empty':
                secondlist=[]
            elif elem == 'power':
                secondlist.append(self.power_number)
            elif elem == 'one':
                secondlist.append(self.pure_numbers[0])
            elif elem == 'two':
                secondlist.append(self.pure_numbers[1])
            else:
                print('bug2', elem)

        return firstlist, secondlist
