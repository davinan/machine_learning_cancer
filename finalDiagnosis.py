import csv
import numpy as np
from numpy import genfromtxt
import numpy.matlib
from numpy import linalg


diagnose = np.genfromtxt('wdbcData.csv', delimiter=',', dtype = None)

Atrain = np.ndarray([455, 31], dtype = float)
Atest = np.ndarray([114, 31], dtype = float)

btrain = np.ndarray([455, 1])
btest = np.ndarray([114, 1])

btrain = btrain.astype( int )
btest = btest.astype ( int )

btrain = btrain - btrain
btest = btest - btest
d = 0
cont = 0
for i in diagnose:
    if cont < 455:
        if i[1] == 'M':
            btrain[d] = 1
    else:
        if i[1] == 'M':
            btest[d-455] = 1
    cont = cont + 1
    d = d+1

n = 0
Atrain = Atrain - Atrain
Atest = Atest - Atest
cont = 0
for y in diagnose:
    for i in range(2, 32):
        if cont < 455:
            Atrain[n][i-2] = Atrain[n][i-2] + y[i]
        else:
            Atest[n-455][i-2] = Atest[n-455][i-2] + y[i]
    n = n+1
    cont = cont + 1
n = 0
cont = 0
for i in range(0, 569):
    if cont < 455:
        Atrain[i][30] = 1
    else:
        Atest[i-455][30] = 1
    cont = cont + 1

A = np.linalg.pinv(Atrain)
coe = np.matmul(A, btrain)

#print coe

Test = np.matmul(Atest, coe)
Test = np.around(Test)

#print Test

wrongs = 0
cont = 0
numB = 0
for i in btest:
    if i != Test[cont]:
        if i == 1:
            if Test[cont] != 2:
                wrongs = wrongs + 1
        else:
            wrongs = wrongs + 1
    cont = cont + 1

acc_score = (float(len(Test)-wrongs)/len(Test))*100
print "Accuracy: ", acc_score
print coe
coe = np.around(coe)
print coe
