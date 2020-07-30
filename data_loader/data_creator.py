import numpy as np
import matplotlib.pyplot as plt
import pickle
# here we create a simple helix data in 3D
set_types = ['E', 'U']
intervals = [[-1, 1], [-2,2]]
steps = [1000, 1000]

from scipy.optimize import root
from scipy import interpolate

def simpleaffine():
    p = 200
    x0 = np.linspace(0, 6, num=p)
    f = 2.2 + 4*x0 + x0**2
    file = open('simple_affine.txt', mode='wb')
    dict = {'n_variables': 1, 'x0': x0, 'f0': f}
    pickle.dump(dict, file)

simpleaffine()

def simplepoly3():
    p = 200
    x0 = np.linspace(0, 6, num=p)
    f = 2.2 + 4*x0 + x0**2 + x0**3
    file = open('simple_poly3.txt', mode='wb')
    dict = {'n_variables': 1, 'x0': x0, 'f0': f}
    pickle.dump(dict, file)

simplepoly3()


simplepoly3()

def simpleoh():
    p = 2000
    t = np.linspace(0, 7.895, num=p)
    f = np.exp(-3*t/2)*(1.167*np.cos(np.sqrt(7)*t/2)+ 3.1*np.sin(np.sqrt(7)*t/2)) + (4*t-3)/16
    # is a solution of f'' = -3f' - 4 f + t
    plt.scatter(t, f)
    plt.show()
    file = open('simple_oscillator.txt', mode='wb')
    dict = {'n_variables': 1, 'x0': t, 'f0': f}
    pickle.dump(dict, file)

simpleoh()

def dummy_diff_vec():
    p = 200
    t = np.linspace(0, 6, num=p)
    x = -0.1 + 12.2 * t + t**2 + t**3
    y = 0.1 + 2.2 * t + t**2+ t**3
    z = -3 + 0.4 * t + t**2+ t**3
    file = open('dummy_diff_vec.txt', mode='wb')
    dict = {'n_variables': 1, 'x0': t, 'f0': x, 'f1': y, 'f2': z}
    pickle.dump(dict, file)

dummy_diff_vec()


def dummy_nodiff_vec():
    p = 200
    t = np.linspace(0, 6, num=p)
    x = -0.1 + 12.2 * t
    y = 0.1 + 2.2 * t
    z = -3 + 0.4 * t
    file = open('dummy_nodiff_vec.txt', mode='wb')
    dict = {'n_variables': 1, 'x0': t, 'f0': x, 'f1' : y, 'f2' : z}
    pickle.dump(dict, file)

dummy_nodiff_vec()

def two_variable_fction():
    p = 200
    x = np.linspace(0, 6, num=p)
    y = np.linspace(0, 6, num=p)
    X, Y = np.meshgrid(x, y, indexing='ij')
    f = 1.2*Y - 0.4*X
    file = open('f_two_variables.txt', mode='wb')
    dict = {'n_variables': 2, 'x0': X, 'x1' : Y, 'f0' : f }
    pickle.dump(dict, file)

two_variable_fction()

def rindler_motion(proper_acc):
    c = 3e8
    p = 200
    # we want acc*t to reach several c :
    u = np.random.randint(1,4)
    tmax = u*c/proper_acc
    t = np.linspace(0, tmax, num=p)
    f = (c**2/proper_acc)*(np.sqrt(1 + (proper_acc*t/c)**2) - 1)
    file = open('Rindler_'+str(proper_acc)+'.txt', mode='wb')
    dict = {'n_variables': 1, 'x0': t, 'f0': f}
    pickle.dump(dict, file)

rindler_motion(20)

def keplerian_motion(eccentricty, v):
    # in cartesian cord : from page 16-19 of https://arxiv.org/pdf/1609.00915.pdf
    # exact motion requires solving a transcendental equation :
    def ff(x,t):
        return x - eccentricty*np.sin(x) - v *t
    p = 5000 #number of points
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
    x = np.squeeze(radius*np.cos(theta))
    y = np.squeeze(radius*np.sin(theta))
    z = np.zeros(p)
    print(x.shape, y.shape, z.shape, t.shape)

    file = open('kepler_1_body.txt', mode='wb')
    dict = {'n_variables': 1, 'x0': t, 'f0': x, 'f1': y, 'f2': z}
    pickle.dump(dict, file)

    #rhs = []
    #diff= []

    #tck = interpolate.splrep(t, x, s=0)
    #f0der = interpolate.splev(t, tck, der=1)
    #f0sec = interpolate.splev(t, tck, der=2)

    #for i in range(len(x)):
    #    rhs.append(-x[i]/(2*(x[i]**2+y[i]**2)**(3/2)))
    #    diff.append((-x[i]/(2*(x[i]**2+y[i]**2)**(3/2)))/f0sec[i])

    #plt.plot(diff)
    #plt.show()

keplerian_motion(0.7,1)

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

def targets_paper_one():
    # --------- one variable targets
    x = np.random.uniform(0, 1, 20)
    x = np.sort(x)
    f = x + x**2 + x**3 + x**4
    file = open('Nguyen2.txt', mode='wb')
    dict = {'n_variables': 1, 'x0': x, 'f0': f, 'maxlen': 20, 'name': 'Nguyen2'}
    pickle.dump(dict, file)

    f = x**6 -2 * x ** 4 + x ** 2
    file = open('Koza3.txt', mode='wb')
    dict = {'n_variables': 1, 'x0': x, 'f0': f, 'maxlen': 15, 'name': 'Koza3'}
    pickle.dump(dict, file)

    f = np.sin(x ** 2) * np.cos(x) - 1
    file = open('Nguyen5.txt', mode='wb')
    dict = {'n_variables': 1, 'x0': x, 'f0': f, 'maxlen': 15, 'name': 'Nguyen5'}
    pickle.dump(dict, file)

    f = np.sin(x + x ** 2) + np.sin(x)
    file = open('Nguyen6.txt', mode='wb')
    dict = {'n_variables': 1, 'x0': x, 'f0': f, 'maxlen': 15, 'name': 'Nguyen6'}
    pickle.dump(dict, file)

    f = x**5 - 2*x**3 + x
    file = open('Koza2.txt', mode='wb')
    dict = {'n_variables': 1, 'x0': x, 'f0': f, 'maxlen': 20, 'name': 'Koza2'}
    pickle.dump(dict, file)

    f = x ** 5 + x**4 + x**3 +  x ** 2 + x
    file = open('Nguyen3.txt', mode='wb')
    dict = {'n_variables': 1, 'x0': x, 'f0': f, 'maxlen': 20, 'name': 'Nguyen3'}
    pickle.dump(dict, file)

    x = np.random.uniform(0, 2, 20)
    x = np.sort(x)
    f = np.log(1+x) + np.log(1+x**2)
    file = open('Nguyen7.txt', mode='wb')
    dict = {'n_variables': 1, 'x0': x, 'f0': f, 'maxlen': 20, 'name': 'Nguyen7'}
    pickle.dump(dict, file)

    x = np.linspace(0, 5, 1+1/0.2)
    f = x + x**2 + x**3 + x**4 + x**5 + x**6
    file = open('Nguyen4.txt', mode='wb')
    dict = {'n_variables': 1, 'x0': x, 'f0': f, 'maxlen': 30, 'name': 'Nguyen4'}
    pickle.dump(dict, file)

    x = np.linspace(0, 6.2, 1+1/0.1)
    f = np.sin(x + x ** 2) + np.sin(x)
    file = open('Sine.txt', mode='wb')
    dict = {'n_variables': 1, 'x0': x, 'f0': f, 'maxlen': 15, 'name': 'Sine'}
    pickle.dump(dict, file)

    x = np.linspace(0, 1, 1+1/0.05)
    f = 0.3*x*np.sin(2*np.pi*x)
    file = open('Keijzer1.txt', mode='wb')
    dict = {'n_variables': 1, 'x0': x, 'f0': f, 'maxlen': 15, 'name': 'Keijzer1'}
    pickle.dump(dict, file)

    x = np.linspace(0, 2, 1+1/0.05)
    f = 0.3 * x * np.sin(2 * np.pi * x)
    file = open('Keijzer2.txt', mode='wb')
    dict = {'n_variables': 1, 'x0': x, 'f0': f, 'maxlen': 15, 'name': 'Keijzer2'}
    pickle.dump(dict, file)

    x = np.linspace(0, 2, 1+1/0.1)
    f = (x+1)**3/(x**2 -x +1)
    file = open('R1.txt', mode='wb')
    dict = {'n_variables': 1, 'x0': x, 'f0': f, 'maxlen': 30, 'name': 'R1'}
    pickle.dump(dict, file)

    f = (x**5 - 3*x**3+1)/(x**2 +1)
    file = open('R2.txt', mode='wb')
    dict = {'n_variables': 1, 'x0': x, 'f0': f, 'maxlen': 30, 'name': 'R2'}
    pickle.dump(dict, file)

    x = np.linspace(0, 1, 1+1/0.05)
    f = (x ** 6 + x ** 5) / (1 + x + x ** 2 + x**3 + x**4)
    file = open('R3.txt', mode='wb')
    dict = {'n_variables': 1, 'x0': x, 'f0': f, 'maxlen': 35, 'name': 'R3'}
    pickle.dump(dict, file)

    x = np.linspace(0, 10, 1+1/0.1)
    f = x ** 3 * np.exp(-x) * np.cos(x) * np.sin(x) * (np.cos(x) * np.sin(x) ** 2 - 1)
    file = open('Keijzer4.txt', mode='wb')
    dict = {'n_variables': 1, 'x0': x, 'f0': f, 'maxlen': 40, 'name': 'Keijzer4'}
    pickle.dump(dict, file)

    x = np.linspace(0, 1, 1+1/0.05)
    f = x + x**2 + x**3 + x**4 + x**5 + x**6+ x**7 + x**8 + x**9
    file = open('Nonic.txt', mode='wb')
    dict = {'n_variables': 1, 'x0': x, 'f0': f, 'maxlen': 40, 'name': 'Nonic'}
    pickle.dump(dict, file)

    # ---------- two variable targets
    x = np.random.uniform(0, 1, 20)
    y = np.random.uniform(0, 1, 20)
    X, Y = np.meshgrid(x, y, indexing='ij')
    f = 8/(2 + X**2 * Y**2)
    file = open('Keijzer14.txt', mode='wb')
    dict = {'n_variables': 2, 'x0': X, 'x1' : Y, 'f0': f, 'maxlen': 20, 'name': 'Keijzer14'}
    pickle.dump(dict, file)

    f = X ** 5 / Y ** 3
    file = open('Meier4.txt', mode='wb')
    dict = {'n_variables': 2, 'x0': X, 'x1': Y, 'f0': f, 'maxlen': 12, 'name': 'Meier4'}
    pickle.dump(dict, file)

    f = np.sin(X) + np.sin(Y ** 2)
    file = open('Nguyen9.txt', mode='wb')
    dict = {'n_variables': 2, 'x0': X, 'x1': Y, 'f0': f, 'maxlen': 12, 'name': 'Nguyen9'}
    pickle.dump(dict, file)

    x = np.random.uniform(0, 3, 20)
    y = np.random.uniform(0, 3, 20)
    X, Y = np.meshgrid(x, y, indexing='ij')
    f = X**4 - X**3 +0.5*Y**2 - Y
    file = open('Keijzer12.txt', mode='wb')
    dict = {'n_variables': 2, 'x0': X, 'x1': Y, 'f0': f, 'maxlen': 30, 'name': 'Keijzer12'}
    pickle.dump(dict, file)

    f = X*Y + np.sin((X-1)*(Y-1))
    file = open('Keijzer11.txt', mode='wb')
    dict = {'n_variables': 2, 'x0': X, 'x1': Y, 'f0': f, 'maxlen': 30, 'name': 'Keijzer11'}
    pickle.dump(dict, file)

    f = X**3/5+Y**3/2 - Y - X
    file = open('Keijzer15.txt', mode='wb')
    dict = {'n_variables': 2, 'x0': X, 'x1': Y, 'f0': f, 'maxlen': 30, 'name': 'Keijzer15'}
    pickle.dump(dict, file)

    x = np.linspace(0.01, 5, 1+1/0.2)
    y = np.linspace(0.01, 5, 1+1/0.2)
    X, Y = np.meshgrid(x, y, indexing='ij')
    f = 1/(1+ 1/X**4) + 1/(1+ 1/Y**4)
    file = open('Pagie1.txt', mode='wb')
    dict = {'n_variables': 2, 'x0': X, 'x1': Y, 'f0': f, 'maxlen': 30, 'name': 'Pagie1'}
    pickle.dump(dict, file)

    x = np.random.uniform(0.3, 4, 20)
    y = np.random.uniform(0.3, 4, 20)
    X, Y = np.meshgrid(x, y, indexing='ij')
    f = np.exp(-(X-1)**2)/(1.2 + (Y-2.5)**2)
    file = open('Vladislavleva1.txt', mode='wb')
    dict = {'n_variables': 2, 'x0': X, 'x1': Y, 'f0': f, 'maxlen': 30, 'name': 'Vladislavleva1'}
    pickle.dump(dict, file)

    x = np.linspace(0.05, 10, 1+1/0.1)
    y = np.linspace(0.05, 10, 5)
    X, Y = np.meshgrid(x, y, indexing='ij')
    f = X**3*np.exp(-X)*np.cos(X)*np.sin(X)*(np.cos(X)*np.sin(X)**2 -1)*(Y-5)
    file = open('Vladislavleva3.txt', mode='wb')
    dict = {'n_variables': 2, 'x0': X, 'x1': Y, 'f0': f, 'maxlen': 45, 'name': 'Vladislavleva3'}
    pickle.dump(dict, file)

    # ---------- three variable targets
    x = np.random.uniform(0, 2, 5)
    y = np.random.uniform(0, 2, 5)
    z = np.random.uniform(1, 5, 10)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    f = 30*X*Z/((X-10)*Y**2)
    file = open('Keijzer5.txt', mode='wb')
    dict = {'n_variables': 3, 'x0': X, 'x1': Y, 'x2': Z, 'f0': f, 'maxlen': 30, 'name': 'Keijzer5'}
    pickle.dump(dict, file)

targets_paper_one()