#  ======================== MONTE CARLO TREE SEARCH ========================== #
# Project:          Symbolic regression tests
# Name:             mcts.py
# Description:      Tentative implementation of a basic MCTS
# Authors:          Adèle Douin & Vincent Reverdy & Jean-Philippe Bruneton
# Date:             2018
# License:          BSD 3-Clause License
# ============================================================================ #

# ================================= PREAMBLE ================================= #
# Packages
import numpy as np
import random
from State import State
from game_env import Game, game_evaluate
import config
import time
# ============================================================================ #

# =============================== CLASS: NODE ================================ #
# A class representing a node of a mcts
class Node:
# ---------------------------------------------------------------------------- #
# Constructs a node from a state
    def __init__(self, voc, state, char, parent = None):

        self.voc = voc
        self.state = state
        self.parent = parent
        self.children = []
        
        self.char = char
        self.proba_children = np.zeros(self.voc.outputdim)

        self.N = 0  # vists
        self.W = 0  # cumulative reward
        self.Q = 0  # average reward


# ---------------------------------------------------------------------------- #
# Checks if the node is a leaf
    def isLeaf(self):
        return len(self.children) == 0

# ---------------------------------------------------------------------------- #
# Checks if the node is a terminal state
    def isterminal(self):
        game = Game(self.voc, self.state)
        return game.isterminal()

# ============================================================================ #



# =============================== CLASS: MCTS ================================ #
# A class representing a mcts
class MCTS:
# ---------------------------------------------------------------------------- #
# Constructs a tree with an optional root
    def __init__(self, maxL, outputdim, voc, player, target, rewardmin, tolerance, root = None):

        self.tolerance = tolerance
        self.player = player
        self.target = target
        self.rewardmin = rewardmin
        self.maxL = maxL
        self.outputdim = outputdim
        self.voc = voc

        #Save terminal states and their rewards
        self.saveterminals = []

# ---------------------------------------------------------------------------- #
# Builds a node from the state and adds it to the tree
    def createNode(self, state, char= None, parent = None):
        node = Node(self.voc, state, char, parent)
        return node
    
# ---------------------------------------------------------------------------- #
#evaluator
    def PUCT(self, child, cpuct) :
        test = True
        if test:
            Q= child.Q
        else:
            if child.isterminal():
                game = Game(self.voc, child.state)
                truereward, _, _ = game_evaluate(game.state.reversepolish, game.state.formulas, self.tolerance, self.voc,
                                                 self.target, "train")
                # print('zz', game.state.reversepolish, truereward)
                Q = truereward
            else:
                Q = child.Q
        P = child.parent.proba_children[child.char -1]
        U = cpuct*P*np.sqrt(child.parent.N)/(1+child.N)
        return Q + U

# ---------------------------------------------------------------------------- #
# Picks a leaf, by first evaluating a function on each node, and picking
# the node where the evaluator is maximal. The evaluator is of the form:
    def pickLeaf(self, node, cpuct):
        random.seed(int(10000000*time.time()%1000))

        #if the provided node is already a leaf (it shall happen only at the first simulation)
        if node.isLeaf():
            return node, node.isterminal()

        else: #the given node is not a leaf, thus pick a leaf descending from the node, according to PUCT :
            current = node
            while current.isLeaf() == False:

                values = []
                count = 0

                for child in current.children:
                    #print(child.state.reversepolish, child.char)
                    values += [self.PUCT(child, cpuct)]
                    count += 1

                maxs = np.argwhere(values == max(values))
             #   if current.state.reversepolish == [4]:
              #      print('here', maxs)
                imax = random.choice(maxs)[0]
                current = current.children[imax]

            # The chosen leaf may be terminal:
        return current, current.isterminal()
# ---------------------------------------------------------------------------- #
# Expand : get all the children of a particular node
    def expandAll(self, leaf):

        game = Game(self.voc, leaf.state)
        allowedchars = game.allowedmoves()

        for char in allowedchars:
            child = self.createNode(game.nextstate(char), char, parent = leaf)
            leaf.children += [child]

# ---------------------------------------------------------------------------- #
# Makes an evaluation with NN of a particular leaf
    def eval_leaf(self, leaf, combo) :

        np.random.seed(int(10000000*time.time()%1000)+int(np.sum(leaf.state.reversepolish)))

        if leaf.isterminal() == 0:

            # call NeuralNet
            nninput = leaf.state.convert_to_NN_input()

            NNreward, NNproba_children = self.player.forward(nninput)
            #print('pc0', proba_children)

            reward = NNreward.detach().numpy()[0][0]
            proba_children = NNproba_children.detach().numpy()[0]

            #if leaf.state.reversepolish == [4]:
            #    print(NNreward, NNproba_children)

            #print('pc', proba_children)
            if config.use_dirichlet and leaf.parent is None :
                probs = np.copy(proba_children)
                alpha = config.alpha_dir
                epsilon = config.epsilon_dir

                dirichlet_input = [alpha for _ in range(self.outputdim)]
                dirichlet_list = np.random.dirichlet(dirichlet_input)
                proba_children = (1 - epsilon) * probs + epsilon * dirichlet_list

            if config.maskinmcts:
                mask = np.zeros(self.outputdim)

                for child in leaf.children:
                    mask[child.char] = 1

                maskit = np.multiply(proba_children, mask)
                leaf.proba_children = maskit / np.sum(maskit)
            else:
                leaf.proba_children = proba_children

            #update
            leaf.W = leaf.W + reward
            leaf.N += 1
            leaf.Q = leaf.W/leaf.N

            if config.use_rd_rollout_in_combo and random.random() < config.rollout_prop and combo:
                self.rollout(leaf)

        else: #we don't have to evaluate the P(a,s)
            if config.usennforterminal:
                nninput = leaf.state.convert_to_NN_input()
                NNreward,_ = self.player.forward(nninput)
                leaf.W = NNreward.detach().numpy()[0][0]
                leaf.N += 1
                leaf.Q = leaf.W / leaf.N
            else:
                game = Game(self.voc, leaf.state)
                truereward, _, _ = game_evaluate(game.state.reversepolish, game.state.formulas, self.tolerance, self.voc, self.target, "train")
                #print('zz', game.state.reversepolish, truereward)
                #if leaf.state.reversepolish == [4,1]:
                 #   print('yo', truereward)
                leaf.W = leaf.W + truereward
                leaf.N += 1
                leaf.Q = leaf.W / leaf.N

# ---------------------------------------------------------------------------- #
# for potential combo method
    def rollout(self, node):
        random.seed(int(10000000*time.time()%1000)+int(np.sum(node.state.reversepolish)))
        #of a non terminal child
        game = Game(self.voc, node.state)

        # Random rollout
        while game.isterminal() == 0:
            nextchar = random.choice(game.allowedmoves())
            game.takestep(nextchar)

        # Propagates the reward
        reward, _, _ = game_evaluate(game.state.reversepolish, game.state.formulas, self.tolerance, self.voc, self.target, "train")

        self.saveterminals.append([reward,game.state.formulas])

        node.W += reward
        node.N += 1
        node.Q = node.W / node.N

# ---------------------------------------------------------------------------- #
# Runs a entire simulation in the tree,

    def simulate(self, node, cpuct, combo):

        leaf, isleafterminal = self.pickLeaf(node, cpuct)

        # the leaf may or may not be terminal. If not terminal :
        if isleafterminal == False:
            self.expandAll(leaf)
            self.eval_leaf(leaf, combo)
            self.backFill(leaf)

        else:
            self.eval_leaf(leaf, combo)
            self.backFill(leaf)

                
# ---------------------------------------------------------------------------- #
# Back propagates values after simulations from a leaf
    def backFill(self, leaf):
        # init recursion
        current = leaf
        add_cumulativereward = leaf.Q

        while current.parent is not None:
            current.parent.N += 1
            current.parent.W += add_cumulativereward
            current.parent.Q = current.parent.W / current.parent.N
            # move up
            current = current.parent

# ========================== END MCTS() CLASS ================================ #
