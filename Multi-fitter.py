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
from Targets import Target, Voc
import game_env
import Evaluate_fit
import time
import utils_mainrun
# ============================================================================ #

class multifit():
    def __init__(self, candidate_formula):
        self.formula = candidate_formula
        #self.files = files
        self.results = []
        # faut reopen chaque file et reinit target en mode scalée, ou pas scalée? je dirai la scalée,
        # ainsi le A ne dépend a priori que de acc proper
        # mais en soi tu veux faire ca dans save resuts en fait
        possible_modes = ['find_function', 'find_1st_order_diff_eq', 'find_2nd_order_diff_eq']

        for u in [20, 200, 2000, 20000, 200000]:
            filenames_train = ['data_loader/Rindler_'+str(u)+'.txt', 'scalar', 'no_diff']
            explicit_time_dependence_allowed = True  # this is expert knowledge since the target is \vec{\ddot{F}(t)} = k \vec{x}/\norm{x}^3 with no time dependence
            use_first_derivatives = False  # idem : no dot(f) in the actual solution (this helps)
            look_for = possible_modes[0]
            expert_knowledge = [explicit_time_dependence_allowed, use_first_derivatives]

            utils_mainrun.sanity_check(filenames_train)  # check data format wrt options chosen
            calculus_mode = filenames_train[-2]

            n_variables, all_targets_name, train_targets = utils_mainrun.init_targets(filenames_train)

            this_target = train_targets[0]
            rangex, rangef = this_target[5]
            maximal_size = 22  # the tree representing the equation in postfix has maximal_size nodes (leaves or internal)

            params = utils_mainrun.init_parameters(n_variables, this_target, all_targets_name, look_for, calculus_mode,
                                     maximal_size, u, expert_knowledge)

            poolsize, delete_ar1_ratio, delete_ar2_ratio, extend_ratio, p_mutate, p_cross, addrandom, voc, \
            bin_length, bin_nscalar, bin_functions, bin_depth, bin_power, bin_target, bin_derivative, bin_var, \
            bin_norm, bin_cross, bin_dot_product = params
            scalar_numbers, alla, rms = game_env.game_evaluate([1], self.formula, voc, train_targets, 'train', 0, look_for)
            rescaled_formula = f.replace('x0', 'x0/'+str(rangex))
            rescaled_formula = str(rangef) + '*(' + rescaled_formula+')'
            self.results.append([rms, alla, rescaled_formula])

            print('donne:', scalar_numbers, alla, rms)
            time.sleep(4.1)

f = 'A*(np.sqrt(1+ A*x0**2) -1)'
f = 'A*np.sqrt(A + x0**A)-A'
f = 'A*np.sqrt(A + A*x0**2)-A'
f = 'A*(np.sqrt(1 + A*x0**2)-1)'
#faut le faire sur le descale pour voir des vars en fait attention x0 est un temps ici reelement
# done
# c'est que y a l'ad
# dans ts les ca, le result sera x = f(t) avec ici  x = L, t = T;  dc CN, x(t) = L*g(t/T); la question étant,
# de combien de parametres physique ou cst fonds peut ou doit dépendre L et T?
#mettons ambitieusement que je l'ignore totalement, on me donne juste une famille, qui eventuellement depend de plusiseurs param physique,
# alors L = L (a, b, c, ..., X, Y, Z) : min sont les param physiques, X etc les cst dimensionnees
# idem pour T
# tu peux pas faire un test d'independence de L et T ici? (reponse en l'occurence, non)
# bah ici L = c^2/a ; T = c/a ; donc ici L = c T; mais pour l'établir rellement faudrait infi de points expé avec a variable dc bon
# en soi c'est impossible je pense c'est a l'expe de dire bah j'ai tourné qu'un seul bouton, dc un seul param physique
# alors si tu dis ca
#L = Lo f(a/ao)
#T = To g (a/ao)
#tes runs montrent justement, donc, que L = c t
#ie
# c (constante numeriquement determinee) =Lo/To * (f/g)(a/ao)
# cad f = g
# dc j'arrive à A*(np.sqrt(1 + A*x0**2)-1) = Lo f(a/ao) * (np.sqrt(1 + (xpourras tt avoir

mf = multifit(f)
res = mf.results
for r in res:
    print(r)