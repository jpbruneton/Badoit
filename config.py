#  ======================== CMA-Based Symbolic Regressor ========================== #
# Project:          Symbolic regression for physics
# Name:             AST.py
# Authors:          Jean-Philippe Bruneton
# Date:             2020
# License:          BSD 3-Clause License
# ============================================================================ #


# _________ define voc ____________
all_fonctions = ['np.cos(', 'np.tan(', 'np.exp(', 'np.log(', 'np.sqrt(', 'np.sinh(', 'np.cosh(', 'np.tanh(', 'np.arcsin(', 'np.arctan(']
#small_set = ['np.sqrt(', 'np.exp(', 'np.log(', 'np.arctan(', 'np.tanh(']

fonctions = all_fonctions # make your choice here

list_scalars = ['one', 'two'] #if running only on some integers

operators = ['+', '-', '*', '/']

#____________termination stuff_______________
iterationa = 5
iterationnoa = 5
termination_nmrse = 3e-8

loglog = False #fit loglog instead of the given target, default False

#_________________Taget related_______________#

# how many nested functions I authorize
MAX_DEPTH = 1
# power is taken only to a real number : avoid stuff like exp(x)^(x exp(x)) !!
only_scalar_in_power = True
fromfile = False
multiple_training_targets = True

if fromfile:
    training_target_list = ['GWfiles\wh_2.txt', 'GWfiles\wl_2.txt']
    maxsize = 15
else:
    training_target_list = 0

# _______________ QD algo related ______________
auto_extent_size = False
add_random = True
skip_no_a_part = False
loadexistingmodel = False
use_simplif = False

# ____________ schedulder _______________
import numpy as np
def get_size(iteration):
    internal_nodes = [15]
    programmation = [100]
    cumuprog = np.cumsum(programmation)
    index = np.where(iteration <= cumuprog)[0]
    index = index[0]
    size = 2*internal_nodes[index]+1
    return size

verifonegivenfunction = False
minrms = 10000 # do not update the grid if rms is greater than this value
qd_init_pool_size = 40
extendpoolfactor = 1.5
#which_target = 'fig1-waveform-H_phase2_1.txt'
smallgrid = False
force_terminal = 0.2

tworunsineval = False
popsize = '10'
timelimit = '7*N'

# reward decreases if there are too many pure scalar parameters, given by:
parsimony = 40
parsimony_cost = 0.01

# max number of scalars allowed in a rd eq:
max_A_number = 1200

# simplif on the fly

# -------------------- reward related -------------------------- #
#after some tests, its better to use both the distance cost AND the derivative cost
usederivativecost = 0  #or 0

#misc
uselocal = True
cpus = 8

specialgal = False
extend_using_smooth_interpolate = False