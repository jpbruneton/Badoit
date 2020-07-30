import numpy as np
from numpy import linalg as la



f = '1*x*(x**2*(1/x)**2 - 1*x**2)**2'
f = f.replace('f(x)', 'f')
f = f.replace('df', 'g')
print(f)
#x = -0.1 + 12.2 * t
#y = 0.1 + 2.2 * t
#z = -3 + 0.4 * t
from sympy import *
t, x, f, g , y= symbols('t x f g y')
print(expand(simplify(1*x*(x**2*(1/x)**2 - 1*x**2)**2)))
x = np.linspace(0,2, 100)
def fx(x):
    return -1.01411910884742*x**3 + 1.4776323356726*x**2*(x + 0.118251301499467)**2.760335 - 0.0161221643985475*x**2 - 1.46921469915072*x*(x + 0.118251301499467)**2.760335 + 1.01947543099771*x - 0.000842539302649056
def f2(x):
    return  x**5 - 2*x**3 + x

import matplotlib.pyplot as plt
plt.plot(fx(x))
plt.plot(f2(x))
plt.show()