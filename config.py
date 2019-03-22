cpus = 40

# --------------  Taget related ---------------- #
findtolerance = True

# --------------------- SPECIAL SYNTAXIC RULES -------------- #
# max lenght
SENTENCELENGHT = 15
# how many nested functions I authorize
MAX_DEPTH = 1
# power is taken only to a real number : avoid stuff like exp(x)^(x exp(x)) !!
only_scalar_in_power = True


tworunsineval = True
popsize = '20*N'
timelimit = '7*N'

# reward decreases if there are too many pure scalar parameters, given by:
parsimony = 200
parsimony_cost = 0.01

# max number of scalars allowed in a rd eq:
max_A_number = 260

# simplif on the fly
use_simplif = True

# -------------------- reward related -------------------------- #
#after some tests, its better to use both the distance cost AND the derivative cost
usederivativecost = 1  #or 0

# -------------   important options ------------- #
use_derivatives = False
