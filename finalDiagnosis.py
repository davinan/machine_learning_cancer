import csv
import numpy as np
from numpy import genfromtxt
import numpy.matlib
from numpy import linalg
# Correction Matrix Plot
import matplotlib.pyplot as plt

# program that uses linear algebra to create a model to diagnose breast cancer
# A: matrix with patient (that had a breast cancer node) feature data
# x: coefficient vector
# b: vector giving the diagnosis of the breast cancer node (1 = malign, 0 = benign)

# get feature data from the dataset
diagnose = np.genfromtxt('wdbcData.csv', delimiter=',', dtype = None, encoding='')

# note: training matrices/vectors take 80% of the data and testing matrices/vectors 20%
# training: (455 patient's breast cancer node features)
# testing: (114 patient's breast cancer node features)

# Create a matrix A for training (currently as zero vectors to be defined later)
Atrain = np.ndarray([455, 31], dtype = float)
Atrain = Atrain - Atrain
# Create a matrix A' for testing (currently as zero vectors to be defined later)
Atest = np.ndarray([114, 31], dtype = float)
Atest = Atest - Atest

# Create a vector b for training
btrain = np.ndarray([455, 1])
btrain = btrain.astype( int )

# Create a vector b' for testing trained model
btest = np.ndarray([114, 1])
btest = btest.astype ( int )

# set both b and b' as zero vectors
# semantically: set both vectors as if all breast cancer nodes are benign
btrain = btrain - btrain
btest = btest - btest

d = 0
cont = 0
# setting the b and b' vectors
for i in diagnose:
    # set the training b vector entries that are supposed to be malign as malign
    if cont < 455 and i[1] == 'M':
        btrain[d] = 1
    # set the testing b vector entries that are supposed to be malign as malign
    elif i[1] == 'M':
        btest[d-455] = 1
    cont = cont + 1
    d = d + 1

n = 0
cont = 0
# setting the A and A' matrices
for y in diagnose:
    # features are in the columns 2 to 31 in the dataset
    for i in range(2, 32):
        if cont < 455:
            Atrain[n][i-2] = Atrain[n][i-2] + y[i]
        else:
            Atest[n-455][i-2] = Atest[n-455][i-2] + y[i]
    n = n + 1
    cont = cont + 1

n = 0
cont = 0
# todo: not sure why I did this...
# for i in range(0, 569):
#     if cont < 455:
#         Atrain[i][30] = 1
#     else:
#         Atest[i-455][30] = 1
#     cont = cont + 1

# get the inverse of matrix A for training
Ainv = np.linalg.pinv(Atrain)
# get the coefficient vector x from the inverse matrix of A (solve for Ax = b)
x = np.matmul(Ainv, btrain)

# multiply the coefficient vector x to the testing matrix A'
test_diagnosis = np.matmul(Atest, x)
test_diagnosis = np.around(test_diagnosis)

wrongs = 0
cont = 0
numB = 0
for i in btest:
    print ("My diagnosis: " + str(test_diagnosis[cont]) + ", real diagnosis " + str(i))
    if i != test_diagnosis[cont]:
        if i == 1:
            if test_diagnosis[cont] != 2:
                wrongs = wrongs + 1
        else:
            wrongs = wrongs + 1
    cont = cont + 1

acc_score = (float(len(test_diagnosis)-wrongs)/len(test_diagnosis))*100
print "Accuracy: ", acc_score
coe = np.around(x, decimals=0)
print "coefficients: "
print coe
