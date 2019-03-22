#  ======================== MONTE CARLO TREE SEARCH ========================== #
# Project:          Symbolic regression tests
# Name:             State.py
# Description:      Tentative implementation of a basic MCTS
# Authors:          Vincent Reverdy & Jean-Philippe Bruneton
# Date:             2018
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

    def __init__(self, voc, state):

        self.voc = voc
        self.reversepolish = state
        self.formulas = self._convert_rpn_to_formula()

# ---------------------------------------------------------------------------- #
# read the rpn vector of equation number k and apply one simplification according to rules given in config

    def one_simplif(self):

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

        for number in self.reversepolish:

            #get character
            char = self.voc.numbers_to_formula_dict[str(number)]

            if number in self.voc.arity0symbols:
                #push scalar in the stack
                stack.append(char)

            elif number in self.voc.arity1symbols:
                sentence = stack[-1]
                newstack = char + sentence + ')'
                if len(stack) == 1:
                    stack = [newstack]
                else:
                    stack = stack[:-1] + [newstack]

            elif number in self.voc.arity2symbols:
                #for not too many useless parenthesis
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

            elif number == self.voc.true_zero_number:
                stack += char

            elif number == self.voc.neutral_element:
                stack += char

            elif number == self.voc.infinite_number:
                stack += char

        #might happen if first symbol is 1 ('halt')
        if len(stack) == 0:
            formula = ''
        else:
            formula = stack[0]


        return formula


# =============================== END CLASS: State ================================ #
