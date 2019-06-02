from game_env import Game
from State import State
import numpy as np
import random
import config
import copy
import game_env


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

        L = len(state.reversepolish)
        if L <= 1:
            return False, state

        #else
        prev_rpn = copy.deepcopy(state.reversepolish)

        if state.reversepolish[-1] == 1:
            char_to_mutate = np.random.randint(0, L - 1)
        else:
            char_to_mutate = np.random.randint(0, L)

        char = prev_rpn[char_to_mutate]

        # ------ arity 0 -------
        if char in self.voc.arity0symbols or char == self.voc.neutral_element or char == self.voc.true_zero_number:
            newchar = random.choice(tuple(x for x in self.voc.arity0symbols if x != char))

        # ------ arity 1 -------
        elif char in self.voc.arity1symbols:
            newchar = random.choice(tuple(x for x in self.voc.arity1symbols if x != char))

        # ------ arity 2 -------
        elif char in self.voc.arity2symbols:
            newchar = random.choice(tuple(x for x in self.voc.arity2symbols if x != char))

        else:
            print('bugmutation', state.reversepolish, char_to_mutate, char)
            raise ValueError


        # --------  finally : I mutate or simply delete the char if -------
        if random.random() < self.delete_ar1_ratio and char in self.voc.arity1symbols:
            #print('bf',state.formulas)
            newrpn = prev_rpn[:char_to_mutate] + prev_rpn[char_to_mutate + 1:]
            newstate = State(self.voc, newrpn)
            #print('af', newstate.formulas)
        else:  # or mutate
            prev_rpn[char_to_mutate] = newchar
            newstate = State(self.voc, prev_rpn)

        # -------- return the new state:

        if self.usesimplif:
            newstate = game_env.simplif_eq(self.voc, newstate)
#            game = Game(self.voc, state)
            #print('bf', game.state.formulas)
 #           game.simplif_eq()
            #print('simplif?')
  #          newstate = game.state
            #print('af', state.formulas)

        # mutation can lead to true zero division (after simplif) thus :
        if self.voc.infinite_number not in newstate.reversepolish :
            return True, newstate
        else:
            return False, state



    # ---------------------------------------------------------------------------- #
    def crossover(self, state1, state2):

        # here i make only crossovers between eqs1 resp. and eqs 2

        prev_state1 = copy.deepcopy(state1)
        prev_state2 = copy.deepcopy(state2)

        game1 = Game(self.voc, prev_state1)
        game2 = Game(self.voc, prev_state2)
        #if game1.getnumberoffunctions()>config.MAX_DEPTH:
        #    print('ici', game1.state.formulas)

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
        start = 2 #+ len(self.voc.arity0symbols)

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
                return False, prev_state1, prev_state2


        # else cant crossover
        else:
            return False, prev_state1, prev_state2

        #returns the new states
        state1 = State(self.voc, rpn1)
        state2 = State(self.voc, rpn2)

        if self.usesimplif:
            state1 = game_env.simplif_eq(self.voc, state1)
            state2 = game_env.simplif_eq(self.voc, state2)

            # game1 = Game(self.voc, state1)
            # game1.simplif_eq()
            # state1 = game1.state

            # game2 = Game(self.voc, state2)
            # game2.simplif_eq()
            # state2 = game2.state
        game1 = Game(self.voc, state1)
        game2 = Game(self.voc, state2)
        toreturn = []

        #crossover can lead to true zero division thus :
        if self.voc.infinite_number in state1.reversepolish :
            toreturn.append(prev_state1)
            #print('fail')

        # also, if it returns too many nested functions, i dont want it (sort of parsimony)
        elif game1.getnumberoffunctions() > config.MAX_DEPTH :
            toreturn.append(prev_state1)
            #print('fail')
        else:
            toreturn.append(state1)
            #print('succes')

        if self.voc.infinite_number in state2.reversepolish:
            toreturn.append(prev_state2)
            #print('fail')

        elif game2.getnumberoffunctions() > config.MAX_DEPTH :
            toreturn.append(prev_state2)
            #print('fail')

        else:
            toreturn.append(state2)
            #print('succes')


        return True, toreturn[0], toreturn[1]

