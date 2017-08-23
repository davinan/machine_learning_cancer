import csv
import numpy as np
from numpy import genfromtxt
import numpy.matlib
from numpy import linalg

prognostic = np.genfromtxt('wpbcData.csv', delimiter=',', dtype = None)

Atrain = np.ndarray([158, 33], dtype = float)
Atest = np.ndarray([40, 33], dtype = float)

btrain = np.ndarray([158, 1])
btest = np.ndarray([40, 1])

btrain = btrain.astype(int)
btest = btest.astype (int)

btrain = btrain - btrain
btest = btest - btest

column = np.ndarray([33, 158])
column2 = np.ndarray([34, 158])

wrongcolumn = np.ndarray([33, 1])

d = 0
cont = 0
for i in prognostic:
    if cont < 158:
        if i[1] == 'N':
            btrain[d] = 1
    else:
        if i[1] == 'N':
            btest[d-158] = 1
    cont = cont + 1
    d = d+1
#print btrain
n = 0
Atrain = Atrain - Atrain
Atest = Atest - Atest
cont = 0
for y in prognostic:
    for i in range(2, 34):
        if cont < 158:
            if y[i] == '?':
                Atrain[n][i - 2] = 0
            else:
                Atrain[n][i - 2] = y[i]#Atrain[n][i - 2] + y[i]
        else:
            if y[i] == '?':
                Atest[n-158][i - 2] = 0
            else:
                Atest[n - 158][i - 2] = y[i]#Atest[n - 158][i - 2] + y[i]
    n = n+1
    cont = cont + 1
n = 0
cont = 0
for i in range(0, 198):
    if cont < 158:
        Atrain[i][32] = 1
    else:
        Atest[i-158][32] = 1
    cont = cont + 1
#print Atrain
cont = 0
for i in range(0, 33):
    for y in Atrain:
        column[i][cont] = y[i]
        cont = cont + 1
    cont = 0

for i in range(0, 33):
    for y in Atest:
        column2[i][cont] = y[i]
        cont = cont + 1
    cont = 0
#print btrain

for i in range(0,158):
    if btrain[i][0] < 1:
        btrain[i][0] = 0

for i in range(0, 33):
    #print column[i]
    feat = column[i]
    #feat1 = np.linalg.pinv(feat)
    coe = np.matmul(feat, btrain)
    best = column2[i]*coe
    #print best
    #print best
    cont2 = 0
    wrongcolumn[i]=0
    best = np.around(best)
    for y in btest:
        if y[0] != best[cont2]:
            if y == 1:
                if best[cont2] != 2:
                    wrongcolumn[i] = wrongcolumn[i] + 1
            else:
                wrongcolumn[i] = wrongcolumn[i] + 1
        cont2 = cont2 + 1

print wrongcolumn
print len(wrongcolumn)