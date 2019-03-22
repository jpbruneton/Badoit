from game_env import Game
from State import State
import numpy as np
import random
import config
import copy



# ===================================================================================#
# this class generates new states from previous states by mutation, crossovers, and other(? #todo)
# takes one or two states and returns one or two states
# deepcopy required since state.reversepolish is a list : mutable

class generate_offsprings():
    def __init__(self, delete_ar1_ratio, p_mutate, p_cross, maximal_size, voc, maxL):
        self.usesimplif = config.use_simplif
        self.p_mutate = p_mutate
        self.delete_ar1_ratio = delete_ar1_ratio
        self.p_cross = p_cross
        self.maximal_size = maximal_size
        self.voc = voc
        self.maxL = maxL


    # ---------------------------------------------------------------------------- #
    # mutation
    def mutate(self, state):

        newstate = []
        prev_state = copy.deepcopy(state.reversepolish)

        #in case mutation gives infinities (see below):
        prev_state_object = copy.deepcopy(state)

        # if I have two equations, then i'll mutate them both:

        L = len(state.reversepolish)

        if state.reversepolish[-1] == 1:
            char_to_mutate = np.random.randint(0, L - 1)
            real_L = L-1
        else:
            char_to_mutate = np.random.randint(0, L)
            real_L = L

        char = prev_state[char_to_mutate]

        # ------ arity 0 -------
        if char in self.voc.arity0symbols or char == self.voc.neutral_element or char == self.voc.true_zero_number:
            newchar = random.choice(tuple(x for x in self.voc.arity0symbols if x != char))

        # ------ arity 1 -------
        elif char in self.voc.arity1symbols:
            newchar = random.choice(tuple(x for x in self.voc.arity1symbols_no_diff if x != char))

        # ------ arity 2 -------
        elif char in self.voc.arity2symbols:
            newchar = random.choice(tuple(x for x in self.voc.arity2symbols if x != char))

        else:
            print('bugmutation', state.reversepolish, char_to_mutate, char)
            raise ValueError


        # --------  finally : I mutate or simply delete the char if -------
        if random.random() < self.delete_ar1_ratio and char in self.voc.arity1symbols:
            newstate.append(prev_state[:char_to_mutate] + prev_state[char_to_mutate + 1:])
        else:  # or mutate
            prev_state[char_to_mutate] = newchar

        # -------- return the new state:
        state = State(self.voc, prev_state)


        if self.usesimplif:
            game = Game(self.voc, self.maxL, state)
            game.simplif_eq()
            state = game.state

        # mutation can lead to true zero division (after simplif) thus :
        if self.voc.infinite_number not in state.reversepolish :
            return state
        else:
            return prev_state_object



    # ---------------------------------------------------------------------------- #
    def crossover(self, state1, state2):

        # here i make only crossovers between eqs1 resp. and eqs 2

        prev_state1 = copy.deepcopy(state1)
        prev_state2 = copy.deepcopy(state2)

        game1 = Game(self.voc, self.maxL, prev_state1)
        game2 = Game(self.voc, self.maxL, prev_state2)

        ast1 = game1.convert_to_ast()
        ast2 = game2.convert_to_ast()

        rpn1 = prev_state1.reversepolish
        rpn2 = prev_state2.reversepolish

        # throw the last '1' == halt if exists:
        if rpn1[-1] == 1:
            array1 = np.asarray(rpn1[:-1])
        else:
            array1 = np.asarray(rpn1)

        if rpn2[-1] == 1:
            array2 = np.asarray(rpn2[:-1])
        else:
            array2 = np.asarray(rpn2)

        # topnode has the max absolute label, so you dont want it/ you want only subtrees, hence the [:-1]
        # subtrees can be scalars == leaves, hence >= 2
        start = 2 + len(self.voc.arity0symbols)

        #get all topnodes of possible subtrees
        positions1 = np.where(array1 >= start)[0][:-1]
        positions2 = np.where(array2 >= start)[0][:-1]

        if positions1.size > 0 and positions2.size > 0:
            #choose two
            which1 = np.random.choice(positions1)
            which2 = np.random.choice(positions2)

            getnonleafnode1 = which1 + 1
            getnonleafnode2 = which2 + 1

            #get the nodes
            node1 = ast1.from_ast_get_node(ast1.topnode, getnonleafnode1)[0]
            node2 = ast2.from_ast_get_node(ast2.topnode, getnonleafnode2)[0]

            #swap parents and children == swap subtrees
            prev1 = node1.parent
            c = 0
            for child in prev1.children:
                if child == node1:
                    prev1.children[c] = node2
                c += 1

            c = 0
            prev2 = node2.parent
            for child in prev2.children:
                if child == node2:
                    prev2.children[c] = node1
                c += 1

            #get the new reversepolish:
            rpn1 = ast1.from_ast_to_rpn(ast1.topnode)
            rpn2 = ast2.from_ast_to_rpn(ast2.topnode)


            # but dont crossover at all if the results are eqs longer than maximal_size (see GP_QD) :
            if len(rpn1) > self.maximal_size or len(rpn2)> self.maximal_size:
                return prev_state1, prev_state2

        # else cant crossover
        else:
            return prev_state1, prev_state2

        #returns the new states
        state1 = State(self.voc, rpn1)
        state2 = State(self.voc, rpn2)

        if self.usesimplif:
            game = Game(self.voc, self.maxL, state1)
            game.simplif_eq()
            state1 = game.state

            game = Game(self.voc, self.maxL, state2)
            game.simplif_eq()
            state2 = game.state

        #crossover can lead to true zero division thus :
        if self.voc.infinite_number not in state1.reversepolish  and self.voc.infinite_number not in state2.reversepolish :
            return state1, state2
        else:
            return prev_state1, prev_state1
