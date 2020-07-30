import numpy as np

data = np.genfromtxt('hubbledat.txt', names=True, dtype=None)
print(data.dtype.names)

import matplotlib.pyplot as plt
from matplotlib import style

style.use('seaborn-colorblind')

#plt.plot(data['zcmb'], data['mb'] + 19.4, '.')
#plt.xscale('log')
#plt.xlim(0.01, 1)
#plt.show()

def DM2DL(DM):
    return 10**(DM/5-1)/1e4

z = data['zcmb']
DL = DM2DL(data['mb']+19.4)

idx   = np.argsort(z)
z = np.array(z)[idx]
DL = np.array(DL)[idx]

#plt.xlabel(r'$D_{L}\;\mathrm{[Mpc]}$',size=18)
#plt.ylabel(r'$z$',size=18)
print(z)
print(DL)
save = np.transpose(np.array([z, DL]))
np.savetxt('dist_luminosite_z.txt', save, delimiter=',')

import pickle
file = open('hubblediagram.txt', mode='wb')
dict = {'n_variables': 1, 'x0': z, 'f0': DL}
pickle.dump(dict, file)

from scipy.integrate import quad
def integrand(t):
    omegam= 0.3
    return 1/np.sqrt((omegam*(1+t)**3 + 1-omegam))

def truelcdm(z):
    c=3*1e8
    mpc = 1
    Ho = 70000/mpc
    I = quad(integrand, 0, z)[0]
    return c*(1 + z)*I/Ho

#z = np.linspace(0,2, num=1000)

def test(x):
    return 1956.91128233032*(x*(0.555031329658823*x**3 - x**2 + 0.000146922104444255*x + 0.481026282844983)/(0.197551935842095*x**3 - 0.35590031468345*x**2 + 0.17121136824))**1.189163 + 22.2517202506795

#res = []
#for thisz in z:
#    res.append(truelcdm(thisz))
#print(res)

from scipy import interpolate
from scipy.interpolate import interp1d
from scipy.optimize import least_squares, curve_fit

def fun(x, a,b,c):
    return a + b*x + c*x**2

xo = [0]*3

print(z.shape, DL.shape)

#remove duplicates:
zfree = []
DLfree = []

c=0
for elem in z:
    if elem not in zfree:
        zfree.append(elem)
        DLfree.append(DL[c])
    c+=1

z = zfree
DL= DLfree


constantsQuad, _ = curve_fit(fun, zfree, DLfree)
print(constantsQuad)

plt.plot(z,DL)
def quadfit(x):
    return constantsQuad[0] + constantsQuad[1]*x + constantsQuad[2]*x*x
def test(x):
    return 17959.4190885226*(0.444468149412413*x)**((1.33463934538819*x**2*(0.444468149412413*x - 0.536061)*np.tan(0.444468149412413*x) + 1176.31308141992*x**2 - 4754.93219686052*np.tan(0.444468149412413*x) + 1113.76701975085)/(971.026177635794*x**2 - 3925.11458808354*np.tan(0.444468149412413*x) + 919.39548156687)) + 43.605469546933

zn = np.linspace(0,3, num=600)
plt.plot(zn,quadfit(zn))

binningz = []
binningdl = []
n = 4
p = 2**n
q = int(len(z)/p)
for i in range(q+1):
    binningz.append(sum(z[i*p: (i+1)*p])/p)
    binningdl.append(sum(DL[i * p: (i + 1) * p])/p)
print(q, i, i*p)
binningz.append(np.mean(z[i*p:]))
binningdl.append(np.mean(DL[i*p:]))
plt.scatter(binningz, binningdl)
#plt.scatter(z, new_f)
plt.show()


