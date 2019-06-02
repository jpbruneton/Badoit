
cpus = 40

# --------------  Taget related ---------------- #
findtolerance = True

# --------------------- SPECIAL SYNTAXIC RULES -------------- #
# how many nested functions I authorize
MAX_DEPTH = 1
# power is taken only to a real number : avoid stuff like exp(x)^(x exp(x)) !!
only_scalar_in_power = True


tworunsineval = False
popsize = '20*N'
timelimit = '7*N'

# reward decreases if there are too many pure scalar parameters, given by:
parsimony = 200
parsimony_cost = 0.01

# max number of scalars allowed in a rd eq:
max_A_number = 260

# simplif on the fly
use_simplif = False

# -------------------- reward related -------------------------- #
#after some tests, its better to use both the distance cost AND the derivative cost
usederivativecost = 1  #or 0


uselocal = False


# -------------------- net and mcts -------------------------- #

data_extension = False #later #todo



use_dirichlet = True
alpha_dir = 0.5
epsilon_dir = 0.15

net = 'resnet' #or 'densenet'
optim = 'sgd'
res_tower = 1
convsize = 64
polfilters=2
valfilters=1
usehiddenpol=False
hiddensize = 64
momentum=0.9
wdecay=0.0001
#----------------#
#NN parameters
LEARNINGRATE=0.001
EPOCHS=4
MINIBATCH = 32
MAXMEMORY = MINIBATCH * 600 #one iteration of 400 games typically creates 600-1000 batches : here we thus save the last 10-6 games or so
MAXBATCHNUMBER = 300 #and we improve the NN by sample randomly in the last maxmemory batches
MINBATCHNUMBER = 64
#----------------#
#MCTS parameters
SIM_NUMBER = 14
CPUCT=1

add_rd_eqs = False
maskinmcts = False
usennforterminal = False
use_rd_rollout_in_combo = True
rollout_prop = 1

loop_self_play = 10
loop_agg_play = 1




