cpus = 40

# --------------  Taget related ---------------- #
findtolerance = True
saveqd = True
use_derivative = True
max_derivative = 2 #will look for f'' -- eventuellement derivees croisees = stuff


extendpoolfactor = 0.001
maxsize = 50
iterationa = 100
which_target = 'fig1-waveform-H_phase2_2.txt'
savedqdpool = 'megapoolphase2_2.txt'
plot = False
# --------------------- SPECIAL SYNTAXIC RULES -------------- #
# how many nested functions I authorize
MAX_DEPTH = 1
# power is taken only to a real number : avoid stuff like exp(x)^(x exp(x)) !!
only_scalar_in_power = True


tworunsineval = False
popsize = '10'
timelimit = '7*N'

# reward decreases if there are too many pure scalar parameters, given by:
parsimony = 1500
parsimony_cost = 0.02

# max number of scalars allowed in a rd eq:
max_A_number = 1200

# simplif on the fly
use_simplif = False

# -------------------- reward related -------------------------- #
#after some tests, its better to use both the distance cost AND the derivative cost
usederivativecost = 0  #or 0

uselocal = True
fromfile = True



