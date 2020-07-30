import csv
import numpy as np
import matplotlib.pyplot as plt

usegal1to6 = False
usestacybin = True
usestacyall = False
if usegal1to6:
    allx = np.array([])
    ally = np.array([])

    alphas= [7.3/7.55,82.4/82.1,6.18/8.66,58.7/47.5,8.11/6.9,3.91/3.78] #div par dist publi
    alphas = [1/x for x in alphas]

    anglepubli=[66.1,38.3,63.7,53.9,39.4,64.5]
    ben_angle = [64, 40, 61, 64, 41, 75]

    uds = [1.01,0.41,0.62,0.48,0.6,0.07]
    ubulge = [1,1,1,1,1,0.93]

    for i in range(6):
        print(i)
        with open('gal' + str(i+1) + '.csv', 'r') as myfile:
            csvReader = csv.reader(myfile)
            opendata= []
            for row in csvReader:
                    opendata.append(row)

        alldata= np.zeros(7)
        for x in opendata:
            x[0] = x[0].replace('\t', ', ')
            res = x[0].split(',')
            line = np.zeros(7)
            for j in range(7):
                line[j] = float(res[j])
            alldata= np.vstack((alldata, line))

        alldata = np.delete(alldata, 0, 0)

        kpc = 3.086*10**(19)
        kms = 1000

        dist = alldata[:, 0]*kpc
        vobs= alldata[:, 1]*kms
        eerv= alldata[:, 2]
        vgas= alldata[:, 3]*kms
        vdisk= alldata[:, 4]*kms
        vbul= alldata[:, 5]*kms
        sbdisk= alldata[:, 6]
        #sbulb= alldata[:, 7]
        print(i)
        alpha = alphas[i]
        u_disk = uds[i]
        u_bulge = ubulge[i]
        angle_correction = np.sin(np.pi*ben_angle[i]/180)/np.sin(np.pi*anglepubli[i]/180)
        ao = 1.2 * (10**(-10))

        v_baryons = np.sqrt(alpha * (u_disk * vdisk**2 + u_bulge * vbul**2 + vgas**2))
        y = (vobs*angle_correction)**2/(dist*ao)

        #corr d'angle
        x = v_baryons**2/(dist*ao)
        allx = np.hstack((allx, x))
        ally = np.hstack((ally, y))


if usestacybin:

    with open('stacy_bin.csv', 'r') as myfile:
        csvReader = csv.reader(myfile)
        opendata = []
        for row in csvReader:
            opendata.append(row)

        alldata = np.zeros(4)
        for x in opendata:
            x[0] = x[0].replace('\t', ', ')
            res = x[0].split(',')
            line = np.zeros(4)
            for j in range(4):
                line[j] = float(res[j])
            alldata = np.vstack((alldata, line))

        alldata = np.delete(alldata, 0, 0)
        print(alldata)
        allx = alldata[:,0]
        ally = alldata[:,1]


if usestacyall:

    with open('stacy_all.csv', 'r') as myfile:
        csvReader = csv.reader(myfile)
        opendata = []
        for row in csvReader:
            opendata.append(row)

        alldata = np.zeros(4)
        for x in opendata:
            x[0] = x[0].replace('\t', ', ')
            res = x[0].split(',')
            line = np.zeros(4)
            for j in range(4):
                line[j] = float(res[j])
            alldata = np.vstack((alldata, line))

        alldata = np.delete(alldata, 0, 0)
        print(alldata)
        allx = alldata[:,0]
        ally = alldata[:,2]



fig = plt.figure()
ax = plt.axes()

#mu = x/(1+x) : le fit depend bcp de ao!!
def nu1(x):
    return (x + np.sqrt(x**2 +4*x))/(2*x)

def test(x):
    return (x*(x + 0.2) + np.log(x + 1))/(x + 0.2)

def test2(x):
    return (x - np.log10((8.871011)))-((x - np.log10((8.871011)))/(((np.exp(x - np.log10((8.871011))))-(x - np.log10((8.871011))))/((0.188739)/(np.exp((0.614191)*(x - np.log10((8.871011))))))))+np.log10((8.871011))


addfake = True

#allx = (10**allx)/(1.2*10**(-10))
#ally= 10**ally/(1.2*10**(-10))
sortx = allx.argsort()
sortedx = allx[sortx][:] + 10
sortedy = ally[sortx][:] + 10
print(len(sortedx))
bins = 50
q = len(sortedx)//bins
binsortex = []
binsortedy = []
for u in range(q):
    binsortex.append(sum(sortedx[u*bins:(u+1)*bins])/bins)
    binsortedy.append(sum(sortedy[u*bins:(u+1)*bins])/bins)

if addfake:
    #sortedx = np.concatenate((np.array([1e-12]), sortedx))
    #sortedy = np.concatenate((np.array([1e-6]), sortedy))
    for u in range(1):
        sortedx = np.concatenate((sortedx, np.array([10+u])))
        sortedy = np.concatenate((sortedy, np.array([10+u])))
    #sortedx = np.concatenate((sortedx, np.array([1000])))
    #sortedy = np.concatenate((sortedy, np.array([1000])))
p = 4.425243
plt.scatter(sortedx[:], sortedy[:])
plt.scatter(sortedx[:], test2(sortedx))
plt.scatter(sortedx[:], sortedx +np.log10(nu1((10**sortedx)/(1.2))) )

plt.show()

#renormalize by bin and smooth it!

if True:
    import pickle
    save_dic = {}
    save_dic['xgal'] = np.asarray(sortedx)
    save_dic['ygal'] = np.asarray(sortedy)

    #save_dic['xgal'] = np.log10(np.asarray(sortedx))
    #save_dic['ygal'] = np.log10(np.asarray(sortedy))

    filename = './galXY.txt'
    with open(filename, 'wb') as file:
        pickle.dump(save_dic, file)

    file.close()

