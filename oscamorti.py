import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt


p = 200
dat = np.zeros((p,2))
trueder = np.zeros(p)
truesec = np.zeros(p)

for i in range(p):
    x = i*6/p
    dat[i,0] = x
    dat[i,1] = 0.8 + np.exp(-x)*(-0.8*np.cos(3*x) +np.sin(3*x)/3)


    trueder[i] = -np.exp(-x)*(-0.8*np.cos(3*x) +np.sin(3*x)/3) + np.exp(-x)*(2.4*np.sin(3*x) +np.cos(3*x))
    truesec[i] = -2*trueder[i]-10*dat[i,1]+8
x = dat[:, 0]
y = dat[:, 1]
tck = interpolate.splrep(x, y, s=0)
yder = interpolate.splev(x, tck, der=1)
ysec = interpolate.splev(x, tck, der=2)
discretder = np.diff(y)/(x[1]-x[0])
discretdersec = np.diff(y, 2)/(x[1]-x[0])**2
print(yder.size)
plt.plot(ysec, 'r')
plt.plot(discretdersec, 'b')
plt.plot(truesec, 'g')
plt.show()

plt.plot(dat)
plt.show()
np.savetxt('oscamorti.txt', dat, delimiter=',')