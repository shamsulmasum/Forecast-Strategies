# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 12:02:09 2020

@author: shamsul
"""
import csv
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import tensorflow as tf
import random as rn
import os
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(40)
rn.seed(1234)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
from keras import backend as K
tf.set_random_seed(123)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

dataset = load_dataset('n.csv')



##########################################################
#### Converting Time Series to Supervised Problem ########
##########################################################

def prepare_dataset(data, step, window_size, H=1, dump=False):
    """ convert time series dataset to supervised dataset """

    # remove rows with None value
    n = data.shape[0] - window_size - H - step + 1

    X = [data[:n]]
    for i in range(1, window_size):
        X.append(data[i:n+i])

    # stack X
    X = np.column_stack(X)

    y = []
    for i in range(window_size, window_size + H):
        h = step + i - 1
        y.append(data[h:n+h])
    y = np.column_stack(y)

    assert y.shape[0] == X.shape[0]

    if dump:
        x_y = np.column_stack([X, y])
        f = x_y.shape[1]
        x_y = scaler.inverse_transform(x_y.reshape(-1, 1)).reshape(-1, f)

        cols = []
        cols.extend(['obs(t+%d)' % (i, ) for i in range(window_size)])
        cols.extend(['pred(t+%d)' % (i + step + window_size - 1, ) for i in range(H)])
        cols = ','.join(cols)

        np.savetxt('dump_n.csv', x_y, header=cols, delimiter=',', fmt='%.0f')

    return X, y

##########################################################
############ Iterative/Recursive Strategy ################
##########################################################

# Different parameters 
nb_epoch = n
n_test = n
n_seq = n

X, y = prepare_dataset(dataset, window_size=n_seq, dump=True)
X = X.reshape(X.shape[0], 1, X.shape[1]) # reshaping X for LSTM

# Model Building

model = Sequential()
model.add(LSTM(10, batch_input_shape=(1, X.shape[1], X.shape[2]), stateful=True))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()


# Splitting Data 
train_x, test_x, train_y, test_y = X[:-n_test], X[-n_test:], y[:-n_test], y[-n_test:]


# Fitting the data to the model
for i in range(nb_epoch):
    model.fit(train_x, train_y, epochs=1, batch_size=1, verbose=1, shuffle=False)
    model.reset_states()

y_pred = []
y_real = train_y[:120]


##########################################################
######## Applying Iterative/Recursive Strategy############
##########################################################


for i in range(120):
    y_pred.append(model.predict(train_x[i].reshape(1, 1, -1)).reshape(-1))
    
    
plt.plot(range(120), y_real)
plt.plot(range(120), y_pred)
plt.xlabel('train accuracy')
plt.show()




##########################################################
################### Direct Strategy ######################
##########################################################


nb_epoch = n
n_test = n
n_seq = n
n_H = 1 


def create_model(dataset, n_step, n_seq):
    """ create a new model with given step """

    X, y = prepare_dataset(dataset, step=n_step, window_size=n_seq)
    X = X.reshape(X.shape[0], 1, X.shape[1])

    print('number of data points:', X.shape[0])
    print('number of features:', X.shape[-1])

    print('features:', X)
    print('Labels:', y)

    # create model
    model = Sequential()
    model.add(LSTM(10, batch_input_shape=(1, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # print model summary
    print('step: %d' % (n_step, ))
    model.summary()

    train_x, train_y = X[:-n_test], y[:-n_test]

    for i in range(nb_epoch):
        model.fit(train_x, train_y, epochs=1, batch_size=1, verbose=1, shuffle=False)
        model.reset_states()

    # print train accuracy
    y_real = train_y[:120]
    y_pred = np.asarray([])

    for i in range(0, 120):
        y_pred = np.append(y_pred, model.predict(train_x[i].reshape(1, 1, -1)).reshape(-1))

    plt.plot(range(120), y_real)
    plt.plot(range(120), y_pred)
    plt.xlabel('train accuracy with step %d' % (n_step, ))
    plt.show()

    return model


# create n_H model each for a different step prediction
models = []
for i in range(n_H):
    models.append(create_model(dataset, i + 1, n_seq))

# create test dataset
X, y = prepare_dataset(dataset, step=1, window_size=n_seq, H=n_H, dump=True)
X = X.reshape(X.shape[0], 1, X.shape[1])

test_x, test_y = X[-n_test:], y[-n_test:]

assert n_test > n_seq

y_real = test_y[:n_test:n_H].reshape(-1)
y_pred = test_y[:n_seq:n_H].reshape(-1)

##########################################################
############### Applying Direct Strategy #################
##########################################################

for i in range(n_seq, n_test, n_H):
    x = y_pred[-n_seq:]
    y = []
    for model in models:
        y.append(model.predict(x.reshape(1, 1, -1)).reshape(-1))
    y_pred = np.append(y_pred, y)

y_pred = y_pred[n_seq:]
y_real = y_real[n_seq:]

print(y_real)



##########################################################
############# Direct Recusive Strategy ###################
##########################################################

nb_epoch = n
n_test = n
n_seq = n
n_H = 1

def create_model(dataset, n_step, n_seq):
    """ create a new model with given step """

    X, y = prepare_dataset(dataset, step=n_step, window_size=n_seq)
    X = X.reshape(X.shape[0], 1, X.shape[1])

    print('number of data points:', X.shape[0])
    print('number of features:', X.shape[-1])

    print('features:', X)
    print('Labels:', y)

    # create model
    model = Sequential()
    model.add(LSTM(10, batch_input_shape=(1, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # print model summary
    print('step: %d' % (n_step, ))
    model.summary()

    train_x, train_y = X[:-n_test], y[:-n_test]

    for i in range(nb_epoch):
        model.fit(train_x, train_y, epochs=1, batch_size=1, verbose=1, shuffle=False)
        model.reset_states()

    # print train accuracy
    y_real = train_y[:120]
    y_pred = np.asarray([])

    for i in range(0, 120):
        y_pred = np.append(y_pred, model.predict(train_x[i].reshape(1, 1, -1)).reshape(-1))

    plt.plot(range(120), y_real)
    plt.plot(range(120), y_pred)
    plt.xlabel('train accuracy with step %d' % (n_step, ))
    plt.show()

    return model

# create n_H model each for a different step prediction
models = []
for i in range(n_H):
    models.append(create_model(dataset, i + 1, n_seq))

# create test dataset
X, y = prepare_dataset(dataset, step=1, window_size=n_seq, H=n_H, dump=True)
X = X.reshape(X.shape[0], 1, X.shape[1])

test_x, test_y = X[-n_test:], y[-n_test:]

assert n_test > n_seq

y_real = test_y[:n_test:n_H].reshape(-1)
y_pred = test_y[:n_seq:n_H].reshape(-1)


##########################################################
######## Applying Direct Recusive Strategy ###############
##########################################################

for i in range(n_seq, n_test, n_H):
    x = y_pred[-n_seq:]
    y = []
    for model in models:
        feed_x = np.append(x, y)[-n_seq:]
        y.append(model.predict(feed_x.reshape(1, 1, -1)).reshape(-1))
    y_pred = np.append(y_pred, y)

y_pred = y_pred[n_seq:]
y_real = y_real[n_seq:]



##########################################################
################### MIMO Strategy ########################
##########################################################


nb_epoch = n
n_test = n
n_seq = n
n_H = n # Greater than 1



X, y = prepare_dataset(dataset, step=1, window_size=n_seq, H=n_H, dump=True)
X = X.reshape(X.shape[0], 1, X.shape[1])

print('number of data points:', X.shape[0])
print('number of features:', X.shape[-1])

print('features:', X)
print('Labels:', y)

# create model
model = Sequential()
model.add(LSTM(10, batch_input_shape=(1, X.shape[1], X.shape[2]), stateful=True))
model.add(Dense(n_H))
model.compile(loss='mean_squared_error', optimizer='adam')

# print model summary
model.summary()

train_x, test_x, train_y, test_y = X[:-n_test], X[-n_test:], y[:-n_test], y[-n_test:]

for i in range(nb_epoch):
    model.fit(train_x, train_y, epochs=1, batch_size=1, verbose=1, shuffle=False)
    model.reset_states()

y_pred = np.asarray([])
y_real = train_y[:120:n_H].reshape(-1)

for i in range(0, 120, n_H):
    y_pred = np.append(y_pred, model.predict(train_x[i].reshape(1, 1, -1)).reshape(-1))

plt.plot(range(120), y_real)
plt.plot(range(120), y_pred)
plt.xlabel('train accuracy')
plt.show()

assert n_test > n_seq

y_real = test_y[:n_test:n_H].reshape(-1)
y_pred = test_y[:n_seq:n_H].reshape(-1)


##########################################################
############## Applying MIMO Strategy ####################
##########################################################

for i in range(n_seq, n_test, n_H):
    x = y_pred[-n_seq:]
    y_pred = np.append(y_pred, model.predict(x.reshape(1, 1, -1)).reshape(-1))



y_pred = y_pred[n_seq:]
y_real = y_real[n_seq:]




##########################################################
################## DIRMO Strategy ########################
##########################################################

nb_epoch = n
n_test = n
n_seq = n
n_H = n # number of data points you want to predict
n_S = n # number of data points each model will predict



def create_model(dataset, n_step, n_seq, n_output):
    """ create a new model with given step """

    X, y = prepare_dataset(dataset, step=n_step, window_size=n_seq, H=n_output)
    X = X.reshape(X.shape[0], 1, X.shape[1])

    print('number of data points:', X.shape[0])
    print('number of features:', X.shape[-1])

    print('features:', X)
    print('Labels:', y)

    # create model
    model = Sequential()
    model.add(LSTM(10, batch_input_shape=(1, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(n_output))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # print model summary
    print('step: %d' % (n_step, ))
    model.summary()

    train_x, train_y = X[:-n_test], y[:-n_test]

    for i in range(nb_epoch):
        model.fit(train_x, train_y, epochs=1, batch_size=1, verbose=1, shuffle=False)
        model.reset_states()

    # print train accuracy
    y_real = train_y[:120:n_output].reshape(-1)
    y_pred = np.asarray([])

    for i in range(0, 120, n_output):
        y_pred = np.append(y_pred, model.predict(train_x[i].reshape(1, 1, -1)).reshape(-1))

    plt.plot(range(120), y_real)
    plt.plot(range(120), y_pred)
    plt.xlabel('train accuracy with step %d' % (n_step, ))
    plt.show()

    return model



assert n_H % n_S == 0   # should be divisble

# we will create n_m MIMO models each will predict n_S outputs
n_m = int(n_H / n_S)

models = []
for i in range(n_m):
    models.append(create_model(dataset, i + 1, n_seq, n_S))

# create test dataset
X, y = prepare_dataset(dataset, step=1, window_size=n_seq, H=n_H, dump=True)
X = X.reshape(X.shape[0], 1, X.shape[1])

test_x, test_y = X[-n_test:], y[-n_test:]

assert n_test > n_seq

y_real = test_y[:n_test:n_H].reshape(-1)
y_pred = test_y[:n_seq:n_H].reshape(-1)


##########################################################
############## Applying DIRMO Strategy ###################
##########################################################


for i in range(n_seq, n_test, n_H):
    x = y_pred[-n_seq:]
    y = []
    for model in models:
        y.append(model.predict(x.reshape(1, 1, -1)).reshape(-1))
    y_pred = np.append(y_pred, y)

y_pred = y_pred[n_seq:]
y_real = y_real[n_seq:]