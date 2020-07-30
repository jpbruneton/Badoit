#  ======================== CMA-Based Symbolic Regressor ========================== #
# Project:          Symbolic regression for physics
# Name:             AST.py
# Authors:          Jean-Philippe Bruneton
# Date:             2020
# License:          BSD 3-Clause License
# ============================================================================ #


# ================================= PREAMBLE ================================= #
# Packages
import copy
# ============================================================================ #


# =============================== CLASS: State ================================ #
# A class representing a state (n equations), as a list of strings or a vector in reverse polish notation (rpn)

class State:
    # ---------------------------------------------------------------------------- #
    # Constructs both the reverse polish vector representations and math formulas

    def __init__(self, voc, state, calculus_mode):

        self.voc = voc
        self.reversepolish = state
        self.calcuusmode = calculus_mode
        self.formulas = self._convert_rpn_to_formula()

    # ---------------------------------------------------------------------------- #
    def one_simplif(self):
        # read the rpn vector and apply one simplification according to simplification rules
        rpn = copy.deepcopy(self.reversepolish)
        change = 0
        index = 0
        while index < len(rpn) and change == 0 :
            sublist=[]
            i = index
            while change == 0 and i < min(index + self.voc.maxrulesize, len(rpn)):
                sublist.append(rpn[i])
                if str(sublist) in self.voc.mysimplificationrules and change == 0:
                    replace = self.voc.mysimplificationrules[str(sublist)]
                    rpn = rpn[:index] + replace + rpn[index + len(sublist) :]
                    change = 1
                i+=1
            index+=1
        return change, rpn

    # ---------------------------------------------------------------------------- #
    def _convert_rpn_to_formula(self):
        #read the RPN from left to right and stack the corresponding string
        stack = []

        # ------------- vectorial case
        if self.calcuusmode == 'vectorial':
            for number in self.reversepolish:
                char = self.voc.numbers_to_formula_dict[str(number)]
                if number in self.voc.arity0symbols:
                    stack.append(char)

                elif number in self.voc.arity1_novec:
                    sentence = stack[-1]
                    newstack = char + sentence + ')' #add parenthesis!
                    if len(stack) == 1:
                        stack = [newstack]
                    else:
                        stack = stack[:-1] + [newstack]

                elif number in self.voc.norm_number:
                    sentence = stack[-1]
                    newstack = char + sentence + ', axis = 1).reshape(SIZE,1)' #reshape required for future evaluation
                    if len(stack) == 1:
                        stack = [newstack]
                    else:
                        stack = stack[:-1] + [newstack]

                elif number in self.voc.arity2novec:
                    if len(stack[-2]) == 1:  # avoid useless parenthesis
                        addleft = stack[-2]
                    else:
                        addleft = '(' + stack[-2] + ')'

                    if len(stack[-1]) == 1:
                        addright = stack[-1]
                    else:
                        addright = '(' + stack[-1] + ')'

                    newstack = stack[:-2] + [addleft + char + addright]
                    stack = newstack

                elif number in self.voc.dot_number: #should read  'np.sum(a * b, axis=1)' + reshape necessary
                    if len(stack[-2]) == 1:
                        addleft = stack[-2]
                    else:
                        addleft = '(' + stack[-2] + ')'

                    if len(stack[-1]) == 1:
                        addright = stack[-1]
                    else:
                        addright = '(' + stack[-1] + ')'

                    newstack = stack[:-2] + ['np.sum(' + addleft + '*' + addright+ ', axis = 1).reshape(SIZE,1)']
                    stack = newstack

                elif number in self.voc.wedge_number: # here its np.cross(a, b)
                    if len(stack[-2]) == 1:
                        addleft = stack[-2]
                    else:
                        addleft = '(' + stack[-2] + ')'

                    if len(stack[-1]) == 1:
                        addright = stack[-1]
                    else:
                        addright = '(' + stack[-1] + ')'

                    newstack = stack[:-2] + [char+addleft+','+ addright+ ')']
                    stack = newstack

                elif number in self.voc.true_zero_number:
                    stack.append(char)

                elif number in self.voc.neutral_element:
                    stack.append(char)

                elif number in self.voc.infinite_number:
                    stack.append(char)

            # might happen if first symbol is 1 ('halt')
            if len(stack) == 0:
                formula = ''
            else:
                formula = stack[0]

            return formula

        # ------------- scalar case
        else:
            for number in self.reversepolish:
                char = self.voc.numbers_to_formula_dict[str(number)]
                if number in self.voc.arity0symbols:
                    stack.append(char)

                elif number in self.voc.arity1symbols:
                    sentence = stack[-1]
                    newstack = char + sentence + ')'
                    if len(stack) == 1:
                        stack = [newstack]
                    else:
                        stack = stack[:-1] + [newstack]

                elif number in self.voc.arity2symbols:
                    if len(stack[-2]) == 1:
                        addleft = stack[-2]
                    else:
                        addleft = '(' + stack[-2] + ')'

                    if len(stack[-1]) == 1:
                        addright = stack[-1]
                    else:
                        addright = '(' + stack[-1] + ')'

                    newstack = stack[:-2] + [addleft + char + addright]
                    stack = newstack

                elif number in self.voc.true_zero_number:
                    stack.append(char)

                elif number in self.voc.neutral_element:
                    stack.append(char)

                elif number in self.voc.infinite_number:
                    stack.append(char)

            if len(stack) == 0:
                formula = ''
            else:
                formula = stack[0]

            return formula

# =============================== END CLASS: State ================================ #

