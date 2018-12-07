import warnings
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

#print tf.VERSION
#print tf.keras.__version__

###############################################################
####################### Build the Model #######################
###############################################################

def build_prognostic_model():
    prog_model = tf.keras.Sequential()
    prog_model.add(layers.Dense(32, activation='relu'))
    prog_model.add(layers.Dense(16, activation=tf.sigmoid))
    #prog_model.add(layers.Dense(32, activation='relu'))
    prog_model.add(layers.Dense(1, activation='softmax'))
    
    #model.compile(optimizer=tf.train.GradientDescentOptimizer, 
    #              loss='categorical_crossentropy'.
    #              metrics=['accuracy'])
    
    # Configure a model for categorical classification.
    prog_model.compile(optimizer=tf.train.GradientDescentOptimizer(0.01),
            loss='mse',#tf.keras.losses.sparse_categorical_crossentropy,
            metrics=['mae'])#tf.keras.metrics.categorical_accuracy])

#       prog_model.compile(optimizer=tf.train.RMSPropOptimizer(0.01),
#                 loss=tf.keras.losses.categorical_crossentropy,
#                 metrics=[tf.keras.metrics.categorical_accuracy])
    return prog_model

###############################################################
######################## LOAD THE DATA ########################
###############################################################

raw_file = np.genfromtxt('wpbcData.csv', delimiter=',', dtype = None) # 198 x 34
labels = np.ndarray([198, 1])
prog_data = np.ndarray([198, 33]) 

count = 0
for row in raw_file:
    if row[1] == 'N':
        labels[count] = 0.9999
    else:
        labels[count] = 0
    count = count + 1

count = 0
for row in raw_file:
    for i in range(2, len(row)):
        prog_data[count][i-2] = row[i] if row[i] != '?' else 0
    count = count + 1

val_labels = labels[155:]
val_data = prog_data[155:]

train_labels = labels[0:155]
train_data = prog_data[0:155]

# print labels 
# print prog_data

###############################################################
############################ TRAIN ############################
###############################################################

model = build_prognostic_model()
print model.weights
model.fit(train_data, train_labels, epochs=32, batch_size=32)
        #validation_data=(val_data, val_labels)) #todo: validation data

print model.weights
