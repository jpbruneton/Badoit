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
from State import State
import config
from Evaluate_fit import Evaluatefit
from AST import AST, Node
import numpy as np
import copy


# =============================== CLASS: Game ================================ #

class Game:

    # ---------------------------------------------------------------------------- #
    # init a game with optional state.

    def __init__(self, voc, state = None):

        self.voc = voc
        self.maxL = voc.maximal_size
        if state is None:
            self.stateinit = []
            self.state = State(voc, self.stateinit)
        else:
            self.state = state

    # ---------------------------------------------------------------------------- #
    def scalar_counter(self):
        #says if the current equation is a scalar or not (if counter == 1)
        #later : add arity 3 symbol in case they exist (like integrals)
        counter = 0

        for char in self.state.reversepolish:
            #infinity doesnt count as a scalr since we discard such equations from the start; see elsewhere
            if char in self.voc.arity0symbols or char == self.voc.neutral_element or char == self.voc.true_zero_number:
                counter += 1

            elif char in self.voc.arity2symbols:
                counter -= 1

        return counter

    # ---------------------------------------------------------------------------- #
    def allowedmoves(self):
        #assuming no arity-3 operators for the moment

        current_state_size = len(self.state.reversepolish)
        space_left = self.maxL - current_state_size

        #init : we go upward so we must start with a scalar
        if current_state_size == 0:
            allowedchars = self.voc.arity0symbols

        else:
            #check if already terminated
            if self.state.reversepolish[-1] == self.voc.terminalsymbol or space_left == 0:
                allowedchars = []

            else:
                scalarcount = self.scalar_counter()
                current_A_number =  sum([1 for x in self.state.reversepolish if x in self.voc.pure_numbers])

                # we must terminate
                if space_left == 1:
                    if scalarcount == 1 : #expression is a scalar -> ok, terminate
                        allowedchars = [self.voc.terminalsymbol]

                    else: # scalarcount cant be greater than 2 at that point thanks to the code afterwards:

                        #take care of power specifics
                        if config.only_scalar_in_power:
                            if self.state.reversepolish[-1] in self.voc.pure_numbers:
                                allowedchars = self.voc.arity2symbols
                            else:
                                allowedchars = self.voc.arity2symbols_no_power

                        else:
                            allowedchars = self.voc.arity2symbols

                else:
                    if scalarcount == 1:
                        allowedchars = [self.voc.terminalsymbol]

                        if space_left >= scalarcount + 1:
                            if self.voc.modescalar == 'noA' :
                                allowedchars += self.voc.arity0symbols
                            else:
                                if current_A_number < config.max_A_number:
                                    allowedchars += self.voc.arity0symbols
                                else:
                                    allowedchars += self.voc.arity0symbols_var_and_tar

                        if space_left >= scalarcount:

                            if self.getnumberoffunctions() < config.MAX_DEPTH :
                                allowedchars += self.voc.arity1symbols


                    if scalarcount >= 2:
                        #take care of power specifics
                        if config.only_scalar_in_power :
                            if self.state.reversepolish[-1] in self.voc.pure_numbers:
                                allowedchars = self.voc.arity2symbols
                            else:
                                allowedchars = self.voc.arity2symbols_no_power

                        else:
                            allowedchars = self.voc.arity2symbols

                        if space_left >= scalarcount+1:
                            if self.voc.modescalar == 'noA' :
                                allowedchars += self.voc.arity0symbols
                            else:
                                if current_A_number < config.max_A_number:
                                    allowedchars += self.voc.arity0symbols
                                else:
                                    allowedchars += self.voc.arity0symbols_var_and_tar

                        #same here
                        if space_left >= scalarcount:

                            # also avoid stuff like exp(f)
                            if self.getnumberoffunctions() < config.MAX_DEPTH :
                                allowedchars += self.voc.arity1symbols

        return allowedchars

    # ---------------------------------------------------------------------------- #
    # returns the number of *nested* functions
    def getnumberoffunctions(self):
        #use the same stack as everywhere else but only keep in the stack the number of nested functions encountered so far
        stack = []
        for number in self.state.reversepolish:
            if number in self.voc.arity0symbols or number == self.voc.true_zero_number or number == self.voc.neutral_element:
                stack += [0]

            elif number in self.voc.arity1symbols:
                if len(stack) == 1:
                    stack = [stack[0] +1]
                else:
                    stack = stack[0:-1] + [stack[-1] + 1]

            elif number in self.voc.arity2symbols:
                stack = stack[0:-2] + [max(stack[-1], stack[-2])]

        return stack[-1]

    #---------------------------------------------------------------------- #
    def nextstate(self, nextchar):
        ''' Given a next char, produce nextstate WITHOUT ACTUALLY UPDATING THE STATE (it is a virtual move)'''
        nextstate = copy.deepcopy(self.state.reversepolish)
        nextstate.append(nextchar)

        return State(self.voc, nextstate)

    # ---------------------------------------------------------------------------- #
    def takestep(self, nextchar):
        ''' actually take the action = update state to nextstate '''
        self.state = self.nextstate(nextchar)

    # ---------------------------------------------------------------------------- #
    def isterminal(self):
        if self.allowedmoves() == []:
            return 1
        else:
            return 0

# ---------------------------------------------------------------------------- #
    def convert_to_ast(self):

        # only possible if the expression is a scalar, thus: for debug
        if self.scalar_counter() !=1:
            print(self.voc.numbers_to_formula_dict)
            print(self.state.reversepolish)
            print(self.state.formulas)
            print('cant convert a non scalar expression to AST')
            raise ValueError

        stack_of_nodes = []
        count = 1
        for number in self.state.reversepolish:

            #init:
            if stack_of_nodes == []:
                ast = AST(number)
                stack_of_nodes += [ast.onebottomnode]

            else:

                if number in self.voc.arity0symbols or number == self.voc.true_zero_number or number == self.voc.neutral_element:
                    newnode = Node(number, 0, None, None ,count)
                    stack_of_nodes += [newnode]

                elif number in self.voc.arity1symbols:
                    lastnode = stack_of_nodes[-1]
                    newnode = Node(number, 1, [lastnode], None ,count)
                    lastnode.parent = newnode

                    if len(stack_of_nodes) == 1:
                        stack_of_nodes = [newnode]

                    if len(stack_of_nodes) >= 2:
                        stack_of_nodes = stack_of_nodes[:-1] + [newnode]

                elif number in self.voc.arity2symbols:
                    newnode = Node(number, 2, [stack_of_nodes[-2], stack_of_nodes[-1]], None, count)
                    stack_of_nodes[-2].parent = newnode
                    stack_of_nodes[-1].parent = newnode
                    stack_of_nodes =  stack_of_nodes[:-2] + [newnode]
            count+=1
        #terminate
        ast.topnode = stack_of_nodes[0]

        return ast



    # ---------------------------------------------------------
    def rename_ai(self, formula):
        scalar_numbers = formula.count('A')

        if scalar_numbers >0:
            neweq = ''
            A_count = 0
            for char in formula:
                if char == 'A':
                    neweq += 'A' + str(A_count)
                    A_count += 1
                else:
                    neweq += char

            return neweq, scalar_numbers

        else:
            return formula, scalar_numbers


# =================  END class Game =========================== #


# ---------------------------------------------------------------------------- #
# create random eqs + simplify it with my rules
def randomeqs(voc):
    game = Game(voc)
    np.random.seed()
    while game.isterminal() == 0:
        nextchar = np.random.choice(game.allowedmoves())
        game.takestep(nextchar)
        #print(game.state.reversepolish, game.state.formulas)

    if config.use_simplif:
        simplestate = simplif_eq(voc, game.state)
        simplegame = Game(voc, simplestate)
        return simplegame
    #print('then', game.state.reversepolish, game.state.formulas)

    else:
        return game

# ---------------------------------------------------------
def simplif_eq(voc, state):
    count = 0
    change = 1

    #print('start simplif', state.reversepolish, state.formulas)
    while change == 1:  # avoid possible infinite loop/ shouldnt happen, but secutity
        change, rpn = state.one_simplif()
        #print('onesimplif', change, rpn)

        state = State(voc, rpn)

        count += 1
        if count > 1000:
            change = 0
    #print('so', state.formulas)
    return state

# ---------------------------------------------------------------------------- #
# takes a non maximal size and completes it with random + simplify it with my rules
def complete_eq_with_random(voc, state):

    if state.reversepolish[-1] == voc.terminalsymbol:
        newstate = State(voc, state.reversepolish[:-1])

    else :
        newstate = copy.deepcopy(state)

    game = Game(voc, newstate)

    while game.isterminal() == 0:
        nextchar = np.random.choice(game.allowedmoves())
        game.takestep(nextchar)

    if config.use_simplif:
        simplestate = simplif_eq(voc, game.state)
        return simplestate
    else:
        return game.state

# -------------------------------------------------------------------------- #
def game_evaluate(rpn, formulas, tolerance, voc, target, mode):

        if voc.infinite_number in rpn:
            return -1, 0, []

        else:
            myfit = Evaluatefit(formulas, voc, target, tolerance, mode)

        return myfit.evaluate()

# -------------------------------------------------------------------------- #
#automatic calculation of tolerance : rd eqs should have mean reward of around 0 (between -1 and 1)
def calculatetolerance(initialguess, target, voc):
    print('automatic calculation of tolerance for reward estimate...')

    budget = 10
    count = 0
    tolerance = initialguess
    meanreward = -1
    print('initial guess', initialguess)

    while (meanreward < -0.8 or meanreward > -0.4) and count < budget:
        meanreward = 0
        p = 20

        for i in range(p):
            game = randomeqs(voc)
            newrdeq = game.state
            if voc.infinite_number not in newrdeq.reversepolish :
                newreward, _, _ =  game_evaluate(game.state.reversepolish, game.state.formulas, tolerance, voc, target, 'train')
                meanreward += newreward
        meanreward /= p

        if meanreward < -0.8:
            tolerance *= np.sqrt(2)
        if meanreward > -0.4:
            tolerance /= np.sqrt(3)
        count+=1
        print(count, meanreward, tolerance)

    if count >= budget -1:
        print('fail to converge')
    print('working with tolerance: ', tolerance, meanreward)
    return tolerance