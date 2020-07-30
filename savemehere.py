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
import time

# =============================== CLASS: Game ================================ #

class Game:

    # ---------------------------------------------------------------------------- #
    # init a game with optional state.
    def __init__(self, voc, state = None):
        self.voc = voc
        self.calculus_mode = voc.calculus_mode
        self.maxL = voc.maximal_size
        if state is None:
            self.stateinit = []
            self.state = State(voc, self.stateinit)
        else:
            self.state = state

    # ---------------------------------------------------------------------------- #
    def scalar_counter(self):
        #says if the current equation is a number (or a vector of numbers!) or not (if counter == 1)
        counter = 0
        for char in self.state.reversepolish:
            #infinity doesnt count as a scalr since we discard such equations from the start; see elsewhere
            if char in self.voc.arity0symbols or char == self.voc.neutral_element or char == self.voc.true_zero_number:
                counter += 1
            elif char in self.voc.arity2symbols:
                counter -= 1
        return counter

    # ------------------------------------------------- #
    def from_rpn_to_critical_info(self):
        can_be_terminated = 0
        for char in self.state.reversepolish:
            if char in self.voc.arity0symbols or char == self.voc.neutral_element or char == self.voc.true_zero_number:
                can_be_terminated += 1
            elif char in self.voc.arity2symbols:
                can_be_terminated -= 1
            else:
                pass

        vec_number = 0 #number of vec in expression
        stack = []  # stack represente les deux dernieres entrees; ordonnées, a entrer dans un op [0,1] est q Avec : attendra un *; mais [1, 0] peut prendre * ou / : ca commute pas
        # on doit faire les deux en meme temps car A + A en vecteur reduit de 1; dc faut savoir si les + ou - agissent sur deux vec ou non

        for char in self.state.reversepolish:
            # arity 0
            if char in self.voc.arity0symbols:
                if char in self.voc.arity0_vec:
                    vec_number+=1
                    stack.append(1)
                else:
                    stack.append(0)

            # arity 1
            elif char in self.voc.arity1symbols:
                if char == self.voc.norm_number:
                    vec_number -= 1
                    if stack[-1] == 1:
                        stack = stack[:-1] + [0]
                    else:
                        print('fixbug: cant take the norm of a scalar')
                        raise ValueError
                # if function like cos : stack doesnt change but for debug:
                else:
                    if stack[-1] != 0:
                        print('cant take cosine of a vector (no pointwise operations allowed by choice)')
                        raise ValueError

            else:  # arity 2
                lasts = stack[-2:]

                if char == self.voc.divnumber:  # can only be [1, 0] : vector divided by scalar gives a vector:
                    if lasts == [1, 0]:
                        toadd = [1]
                    elif lasts == [0, 0]:
                        toadd = [0]
                    else:
                        print('fixbug: scalar cant be divided by vector; or vector by vector')
                        raise ValueError

                elif char == self.voc.multnumber:
                    if lasts == [0, 0]:
                        toadd = [0]
                    elif lasts == [0, 1] or lasts == [1, 0]:
                        toadd = [1]
                    else:
                        print('fixbug: vectors cant be multiplied')
                        raise ValueError

                elif char == self.voc.plusnumber or char == self.voc.minusnumber:
                    if lasts == [0,0]:
                        toadd = [0]
                    elif lasts == [0,1] or lasts == [1,0]:
                        print('fixbug: scalars cant be added to a vector')
                        raise ValueError
                    else: # add two vectors : reduce n_vec one unit
                        toadd = [1]
                        vec_number -=1


                elif char == self.voc.power_number:
                    if lasts == [0, 0]:
                        toadd = [0]
                    else:
                        print('bugfixing : power not authorized here')
                        raise ValueError

                elif char == self.voc.wedge_number:
                    vec_number -= 1
                    if lasts == [1, 1]:
                        toadd = [1]
                    else:
                        print('bug : wedge not allowed')
                        raise ValueError

                elif char == self.voc.dot_number:
                    vec_number -= 2
                    if lasts == [1, 1]:
                        toadd = [0]
                    else:
                        print('bug : dot product not allowed')
                        raise ValueError

                # update stack for case arity 2
                stack = stack[:-2] + toadd


        print('avec le state', self.state.reversepolish, self.state.formulas)
        print('jai', can_be_terminated, vec_number, stack)

        return can_be_terminated, vec_number, stack


    # ---------------------------------------------------------------------------- #
    def allowedmoves_vectorial(self):
        current_state_size = len(self.state.reversepolish)
        space_left = self.maxL - current_state_size
        current_A_number = sum([1 for x in self.state.reversepolish if x == self.voc.pure_numbers[0]])
        current_A_number += sum([3 for x in self.state.reversepolish if x == self.voc.pure_numbers[1]])

        # init : we go upward so we must start with a scalar
        if current_state_size == 0:
            allowedchars = self.voc.arity0symbols #start either with a vec or a scalar

        else:
            # check if already terminated
            if self.state.reversepolish[-1] == self.voc.terminalsymbol or space_left == 0:
                allowedchars = []

            else:
                can_be_terminated, vec_number, stack = self.from_rpn_to_critical_info()
                info = [self.state.formulas, can_be_terminated, vec_number, stack]

                if can_be_terminated == 0:
                    print('bug : shdnot happen at all car l init a eu lieu', info)
                    raise ValueError

                # -----------
                if space_left == 1:
                    if can_be_terminated == 1: # must terminate
                        allowedchars = [self.voc.terminalsymbol]
                        if vec_number !=1:
                            print('this shd not happen sinon expression non vectorielle', info)
                            raise ValueError

                    elif can_be_terminated == 2:
                        if vec_number == 1: #deux nombres mais un seul vec, de type q E necessite * ou /
                            if stack[-2:] == [0, 1]:
                                allowedchars = [self.voc.multnumber]  # si last are q E ; si c'est E q : / autorisé aussi mais enfin ca revient au meme
                            elif stack[-2:] == [1, 0]:
                                allowedchars = [self.voc.multnumber, self.voc.divnumber]
                            elif stack[-2:] == [1, 1]:  # A wedge A, A+A ou A -A
                                allowedchars = [self.voc.wedge_number, self.voc.plusnumber, self.voc.minusnumber]
                            else:
                                print('cas non prevu', info)
                                raise ValueError
                    else:
                        print('bug : cant terminate', info)
                        raise ValueError

                # -------------
                elif space_left == 2 :

                    if can_be_terminated == 1:
                        if vec_number == 0: #must add a vector
                            allowedchars = self.voc.arity0_vec
                        elif vec_number==1:
                            allowedchars = [self.voc.terminalsymbol]
                            allowedchars+= self.voc.arity0symbols
                        else:
                            print('cant happen because can be terminated wd be >1', info)
                            raise ValueError


                    elif can_be_terminated == 2: #op required
                        if vec_number == 0:
                            print('must anticipate ce cas car ici pas terminable')
                            raise ValueError

                        elif vec_number == 1:
                            if stack[-2:] == [0,1]:
                                allowedchars = [self.voc.multnumber]
                            elif stack[-2:] == [1, 0]:
                                allowedchars = [self.voc.multnumber] + [self.voc.divnumber]
                            else:
                                print('bug pas normal', info)
                                raise ValueError

                        elif vec_number == 2: #op required et baisser d'un vec, donc
                            allowedchars = [self.voc.plusnumber, self.voc.minusnumber, self.voc.wedge_number]

                        else:
                            print('cas pas prevu ici?', info)
                            raise ValueError

                    elif can_be_terminated == 3: #2 op required
                        if vec_number == 0:
                            print('must not happen', info)
                            raise ValueError
                        elif vec_number == 1:
                            if stack[-3:] == [0,0,1]:
                                allowedchars = [self.voc.multnumber]
                            elif stack == [0,1,0]:
                                allowedchars = [self.voc.multnumber, self.voc.divnumber]
                            elif stack == [1,0,0]:
                                allowedchars = self.voc.arity2novec #todo check : shd be + - * / power
                            else:
                                print('cas pas prevu ici??', info)
                                raise ValueError
                        elif vec_number == 2: #2 op required et doit eliminer un vecteur : donc:
                            if stack[-3:] == [0,1,1]:
                                allowedchars = [self.voc.plusnumber, self.voc.minusnumber, self.voc.wedge_number] #pas le dot pour ne pas rendre scal l'expression finale
                            elif stack == [1,0,1]:
                                allowedchars = [self.voc.multnumber]
                            elif stack == [1,1,0]:
                                allowedchars = [self.voc.multnumber, self.voc.divnumber]
                            else:
                                print('cas pas prevu ici??', info)
                                raise ValueError

                        elif vec_number == 3: # cas stack 1 1 1
                            allowedchars = [self.voc.dot_number, self.voc.plusnumber, self.voc.minusnumber, self.voc.wedge_number] #mais ici oui

                        else:
                            print('impossible n est ce pas?', info)
                            raise ValueError

                    # -------------
                    elif space_left >= 3: #cas general

                        t = can_be_terminated - 1 # this equals to the number of operators required
                        nu = vec_number -1 # number of extra vectors : if >=1 : must reduce the number of vectors
                        p = space_left

                        if p <= t:
                            print('on ne pouura pas termminer à terme : shd never happen', info)
                            raise ValueError

                        elif p == t: # t operator required dans t space left : only operators allowed here
                            if nu == -1:
                                print('on ne pouura pas termminer à terme avec une expression vectorielle : shd never happen', info)
                                raise ValueError
                            elif nu >= 0:
                                #completion tjs possible : on ajoute tout operatuer compatible avec le stack
                                lasts = stack[-2:]
                                if lasts == [0,0]:
                                    allowedchars = self.voc.arity2novec
                                elif lasts == [0,1]:
                                    allowedchars = [self.voc.multnumber]
                                elif lasts == [1, 0]:
                                    allowedchars = [self.voc.multnumber, self.voc.divnumber]
                                else:
                                    allowedchars = [self.voc.dot_number, self.voc.plusnumber, self.voc.minusnumber, self.voc.wedge_number]

                        elif p == t+1:
                            #pas assez de place pour ajouter un scal cqr il requierera aussi un op, mais les fonctions, oui, y compris la norme si vec es t plus grand strict que 1
                            if stack[-1] == 0:
                                allowedchars = self.voc.arity1_novec
                            elif stack[-1] == 1 and nu >0:
                                allowedchars = [self.voc.norm_number]

                        else : #on peut tout ajouter mais on enforce d'abord un vec si y en a pas
                            if nu==-1:
                                allowedchars  = self.voc.arity1_vec
                            elif nu >= 0:
                                #on peut terminer maybe
                                allowedchars = []
                                if nu == 0 and t == 0:
                                    allowedchars = [self.voc.terminalsymbol]
                                #et dans tous les cas:
                                allowedchars += self.voc.arity0symbols
                                if stack[-1] == 0:
                                    allowedchars+= self.voc.arity1_novec
                                if stack[-1] == 1:
                                    allowedchars += [self.voc.norm_number]

                                lasts = stack[-2:]
                                if lasts == [0, 0]:
                                    allowedchars += self.voc.arity2novec
                                if lasts == [0, 1]:
                                    allowedchars += [self.voc.multnumber]
                                if lasts == [1, 0]:
                                    allowedchars += [self.voc.multnumber, self.voc.divnumber]
                                if lasts == [1,1]:
                                    allowedchars += [self.voc.dot_number, self.voc.plusnumber, self.voc.minusnumber,
                                                    self.voc.wedge_number]

                            #todo implement max depth encore to do

        return allowedchars
    # ---------------------------------------------------------------------------- #
    def allowedmoves_novectors(self):
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

                # check if we must terminate
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
    def getnumberoffunctions(self, state = None):
        #use the same stack as everywhere else but only keep in the stack the number of nested functions encountered so far
        stack = []
        if state is None:
            state = self.state
        for number in state.reversepolish:
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
        if self.calculus_mode == 'scalar':
            if self.allowedmoves_novectors() == []:
                return 1
            else:
                return 0
        if self.calculus_mode == 'vectorial':
            print('checkin terminal, ignore')
            if self.allowedmoves_vectorial() == []:
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
    game = Game(voc, state=None)
    np.random.seed()
    while game.isterminal() == 0:
        if voc.calculus_mode == 'scalar':
            nextchar = np.random.choice(game.allowedmoves_novectors())
            game.takestep(nextchar)
        else:
            allowed_moves = game.allowedmoves_vectorial()
            nextchar = np.random.choice(allowed_moves)
            game.takestep(nextchar)
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
        if voc.calculus_mode == 'scalar':
            nextchar = np.random.choice(game.allowedmoves_novectors())
            game.takestep(nextchar)
        else:
            nextchar = np.random.choice(game.allowedmoves_vectorial())
            game.takestep(nextchar)
    if config.use_simplif:
        simplestate = simplif_eq(voc, game.state)
        return simplestate
    else:
        return game.state

# -------------------------------------------------------------------------- #
def game_evaluate(rpn, formulas, voc, train_targets, mode, u, look_for):
        if voc.infinite_number in rpn:
            return 0, [], 100000000
        else:
            myfit = Evaluatefit(formulas, voc, train_targets, mode, u, look_for)
        return myfit.evaluate()