import csv
import numpy as np
from numpy import genfromtxt
import numpy.matlib
from numpy import linalg
# Correction Matrix Plot
import matplotlib.pyplot as plt

# gets largest absolute valued coefficients in coefficient vectors
# returns vector with largest coefficients
def get_n_largest_abs(n, coe_vector):
    sign_vector = np.copy(coe_vector)
    for i in range(0, len(sign_vector)):
        if (sign_vector[i] < 0):
            sign_vector[i] = -1
        else:
            sign_vector[i] = 1
    n_layers = np.ndarray([n, len(coe_vector)], dtype = float);
    coe_vector = np.absolute(coe_vector)
    for i in range(0, n):
        largest = np.amax(coe_vector)
        for y in range(0, len(coe_vector)):
            if largest != coe_vector[y]:
                n_layers[i][y] = 0
            else:
                n_layers[i][y] = largest
                coe_vector[y] = -1000
    largest_coe = n_layers[0]
    for i in range(1, n):
        largest_coe = np.add(largest_coe, n_layers[i])
    for i in range(1, len(largest_coe)):
        largest_coe[i] = largest_coe[i] * sign_vector[i]
    return largest_coe

# gets largest coefficients in coefficient vectors
# returns vector with largest coefficients
def get_n_largest(n, coe_vector):
    n_layers = np.ndarray([n, len(coe_vector)], dtype = float);
    coe_vector = np.absolute(coe_vector)
    for i in range(0, n):
        largest = np.amax(coe_vector)
        for y in range(0, len(coe_vector)):
            if largest != coe_vector[y]:
                n_layers[i][y] = 0
            else:
                n_layers[i][y] = largest
                coe_vector[y] = -1000
    largest_coe = n_layers[0]
    for i in range(1, n):
        largest_coe = np.add(largest_coe, n_layers[i])
    return largest_coe

def get_acc_score(Atest, btest, x):
    test_diagnosis = np.matmul(Atest, x)
    test_diagnosis = np.around(test_diagnosis)

    test_diagnosis = np.absolute(test_diagnosis)

    wrongs = 0
    cont = 0
    numB = 0
    for i in btest:
        print ("My prognosis: " + str(test_diagnosis[cont]) + ", real prognosis" + str(i))
        if i != test_diagnosis[cont]:
            if i == 1:
                if test_diagnosis[cont] != 2:
                    wrongs = wrongs + 1
            else:
                wrongs = wrongs + 1
        cont = cont + 1

    acc_score = (float(len(test_diagnosis)-wrongs)/len(test_diagnosis))*100
    coe = np.around(x)
    print "Accuracy: ", acc_score
    print "coefficients: "
    print coe
# program that uses linear algebra to create a model to make a prognosis for breast cancer
# A: matrix with patient (that had a breast cancer node) feature data
# x: coefficient vector
# b: vector giving the prognosis of the breast cancer node (1 = recurrent, 0 = not recurrent)

# get feature data from the dataset
data = np.genfromtxt('wpbcData.csv', delimiter=',', dtype = None, encoding=None)

# note: training matrices/vectors take 80% of the data and testing matrices/vectors 20%
# training: (158 patient's breast cancer node features)
# testing: (40 patient's breast cancer node features)

num_train = 158
num_test = 40
num_feat = 33

# Create a matrix A for training (currently as zero vectors to be defined later)
Atrain = np.ndarray([num_train, num_feat], dtype = float)
Atrain = Atrain - Atrain
# Create a matrix A' for testing (currently as zero vectors to be defined later)
Atest = np.ndarray([num_test, num_feat], dtype = float)
Atest = Atest - Atest

# Create a vector b for training
btrain = np.ndarray([num_train, 1])
btrain = btrain.astype( int )

# Create a vector b' for testing trained model
btest = np.ndarray([num_test, 1])
btest = btest.astype ( int )

# set both b and b' as zero vectors
# semantically: set both vectors as if all breast cancer nodes are benign
btrain = btrain - btrain
btest = btest - btest

d = 0
cont = 0
# Observation: this program works better setting nonrecurrent as 1
#              and recurrent as 0. Probably because number of nonrecurrent
#              occurrences is larger than recurrent.
#
# setting the b and b' vectors
for i in data:
    # set the training b vector entries that are supposed to be malign as malign
    if cont < num_train and i[1] == 'N':
        btrain[d] = 1
    # set the testing b vector entries that are supposed to be malign as malign
    elif i[1] == 'N':
        btest[d-num_train] = 1
    cont = cont + 1
    d = d + 1

n = 0
cont = 0
# setting the A and A' matrices
for y in data:
    # features are in the columns 2 to 33 in the dataset
    for i in range(2, num_feat + 1):
        if cont < num_train:
            Atrain[n][i-2] = Atrain[n][i-2] + y[i]
        else:
            Atest[n-num_train][i-2] = Atest[n-num_train][i-2] + y[i]
    n = n + 1
    cont = cont + 1

n = 0
cont = 0

# get the inverse of matrix A for training
for y in data:
    for i in range(2, num_feat+1):
        if cont < num_train:
            if y[i] == '?':
                Atrain[n][i - 2] = 0
            else:
                Atrain[n][i - 2] = y[i]#Atrain[n][i - 2] + y[i]
        else:
            if y[i] == '?':
                Atest[n - num_train][i - 2] = 0
            else:
                Atest[n - num_train][i - 2] = y[i]#Atest[n - 158][i - 2] + y[i]
    n = n+1
    cont = cont + 1

Ainv = np.linalg.pinv(Atrain)

# get the coefficient vector x from the inverse matrix of A (solve for Ax = b)
x = np.matmul(Ainv, btrain)
get_acc_score(Atest, btest, x)

# get fifteen most relevant coeficcients
most_relevant = get_n_largest_abs(15, np.copy(x)) # 82.5% accuracy
get_acc_score(Atest, btest, most_relevant)

# get fourteen most relevant coeficcients
most_relevant = get_n_largest_abs(14, np.copy(x)) # 87.5% accuracy
gget_acc_score(Atest, btest, most_relevant)
