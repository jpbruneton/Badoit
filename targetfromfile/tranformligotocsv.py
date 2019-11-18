import numpy as np
import csv



dat = np.loadtxt('fig1-waveform-H.txt', delimiter=',')

times = dat[:,0]
import matplotlib.pyplot as plt
plt.plot(np.diff(times))
