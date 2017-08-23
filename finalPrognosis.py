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

btrain = btrain.astype( int )
btest = btest.astype ( int )

btrain = btrain - btrain
btest = btest - btest
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

Ainv = np.linalg.pinv( Atrain )
coe = np.matmul(Ainv, btrain)


Test = np.matmul(Atest, coe)
#print Test
Test = np.around(Test)
#print Test
wrongs = 0
cont = 0
for i in btest:
    if i != Test[cont]:
        if i == 1:
            if Test[cont] != 2:
                wrongs = wrongs + 1
        else:
            wrongs = wrongs + 1
    cont = cont + 1
#print Test
acc_score = ((float(len(Test)-wrongs)/len(Test))*100)
print "Accuracy: ", acc_score
#coe = np.around(coe)
print "Coefficients: "
print coe
#print btest
#print btrain
# 1 = nonrecurrent