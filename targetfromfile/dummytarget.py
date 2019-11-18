import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('fig1-waveform-H.txt', delimiter=',')
L = data.shape[0]

start = 1933
end = 3200
trustart = 2.5e-01
trueend = 4.599609375000000000e-01
interval = trueend - trustart
timestart = trustart + start*interval/L
timeend = trustart + end*interval/L
print(timestart, timeend)

data_x = list(data[start:end, 0])
data_f = list(data[start:end,1])
x_train = data_x[0::10]
x_test = data_x[0::7]
f0_train = data_f[0::10]
f0_test = data_f[0::7]

print(len(f0_test), len(f0_train))
plt.plot(f0_train, 'r')
#plt.plot(f0_test, 'b')

x = np.linspace(timestart, timeend, (end-start)/10)
f = ((np.sin((-7.782403)*(x)))-(np.sin((-132.414166)*((((12.415785)-((-12.416299)-(((-8.62986)+((1.003467)/(x)))/(x))))/(x))+(-23.805664)))))/((((np.log((6.453774)))-(np.cos((7.587297)/(x))))/((np.cos((5.638108)/(3.006802)))/(np.cos((3.543609)-(0.401313)))))-(((16.439841)*(np.cos((-0.607409))))-(11.69155)))
plt.plot(f)
plt.show()
