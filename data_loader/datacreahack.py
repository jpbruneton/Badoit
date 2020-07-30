import numpy as np
import matplotlib.pyplot as plt
# here we create a simple helix data in 3D
set_types = ['E', 'U']
intervals = [[-1, 1], [-2,2]]
steps = [1000, 1000]

from scipy.optimize import root
from scipy import interpolate

def simpleoh():
    p = 200
    t = np.linspace(0, 6, num=p)
    y = 2.2*np.exp(-t/4)*np.cos(4.21*t)
    t = list(t)
    data = np.transpose(np.array([t, y]))
    np.savetxt('simpleoh1D.csv', data, delimiter=',')

def keplerian_motion(eccentricty, v):
    # in cartesian cord : from page 16-19 of https://arxiv.org/pdf/1609.00915.pdf
    # exact motion requires solving a transcendental equation :
    def ff(x,t):
        return x - eccentricty*np.sin(x) - v *t
    p=5000
    t = np.linspace(0, 6, num=p)
    sols=[]

    for elem in t:
        def f(x):
            return ff(x,elem)

        sol = root(f, 0) #0 is the initial guess
        sols.append(sol.x)
    theta = []
    for elem in sols:
        theta.append(2*np.arctan(np.tan(elem/2)*np.sqrt((1 + eccentricty)/(1-eccentricty))))

    radius = 1/(1+ eccentricty*np.cos(theta))
    #plt.polar(theta, radius)
    #plt.show()
    x = list(radius*np.cos(theta))
    y = list(radius*np.sin(theta))
    z= [0]*p
    t = list(t)
    data = np.transpose(np.array([t, x, y,z]))
    np.savetxt('kepler_1.csv', data, delimiter=',')
    rhs =[]
    diff=[]

    tck = interpolate.splrep(t, x, s=0)
    f0der = interpolate.splev(t, tck, der=1)
    f0sec = interpolate.splev(t, tck, der=2)

    for i in range(len(x)):
        rhs.append(-x[i]/(2*(x[i]**2+y[i]**2)**(3/2)))
        diff.append((-x[i]/(2*(x[i]**2+y[i]**2)**(3/2)))/f0sec[i])

    #plt.plot(diff)
    #plt.show()

#keplerian_motion(0.7,1)

def easy_target():
    set_types = ['E', 'E']
    intervals = [[-1, 1], [-2, 2]]
    steps = [100, 100]

    target_x = '5*np.cos(2*t)'
    target_y = '1.2*np.cos(2*t -0.568)'
    target_z = '0'

    # first create train function then test functions
    for u in range(2):
        if set_types[u] == 'E':
            t = np.linspace(intervals[u][0], intervals[u][1], num=steps[u])
        elif set_types[u] == 'U':
            t = np.random.uniform(intervals[u][0], intervals[u][1], steps[u])
            t = np.sort(t)


        x_t = eval(target_x)
        y_t = eval(target_y)
        z_t = eval(target_z)
        t= list(t)
        x_t = list(x_t)
        y_t = list(y_t)
        z_t = list(y_t)
        dat = np.transpose(np.array([t, x_t, y_t, z_t]))
        print(dat.shape)
        if u == 0:
            np.savetxt('x1_train(t).csv', dat, delimiter=',')
        else:
            np.savetxt('x1_test(t).csv', dat, delimiter=',')

#easy_target()

def monddata():
    import pickle
    with open('galXY.txt', 'rb') as file:
        load_dic = pickle.load(file)

    x = load_dic['xgal']
    f0 = load_dic['ygal']

    file.close()

    dat = np.transpose(np.array([x, f0]))
    np.savetxt('mond_bin.csv', dat, delimiter=',')


monddata()

def testdimensione():
    p = 2000
    t = np.linspace(0.1, 6, num=p)
    L = 12
    y = L*((t/L)**2 + 0.4*(t/L)-0.7)
    t = list(t)
    plt.plot(y)
    plt.show()
    data = np.transpose(np.array([t, y]))
    np.savetxt('testdim.csv', data, delimiter=',')

