# -*- coding: utf-8 -*-
'''
@Time    : 2021/3/5 18:59
@Author  : yany
@File    : course1_week2.py
'''

import numpy as np
import h5py
import matplotlib.pyplot as plt

# read data
def load_data(path, x_flag, y_flag):
    data_set = h5py.File(path, 'r')
    x_data = np.array(data_set[x_flag])
    y_data = np.array(data_set[y_flag])
    len_y = y_data.shape[0]
    y_data = y_data.reshape((1, len_y))
    return x_data, y_data


train_x, train_y = load_data('./raw_data/1_2/train_catvnoncat.h5', 'train_set_x', 'train_set_y')
test_x, test_y = load_data('./raw_data/1_2/test_catvnoncat.h5', 'test_set_x', 'test_set_y')
# look the picture
# plt.imshow(train_x[24])

train_shape = train_x.shape
test_shap = test_x.shape

# data condition
print('amount of train: m_train = ', train_shape[0])
print('amount of test : m_test = ', test_shap[0])
print('picture info :', train_shape[1:])
print('train_x dimension is:', train_shape)
print('train_y dimension is:', train_y.shape)
print('test_x dimension is', test_shap)
print('test_y dimension is', test_y.shape)

# transfrom the x data's shape to [x(1), x(2), x(3), ..., x(n)]
# flatten
def flatten_func(data):
    shape_ = data.shape
    return data.reshape(shape_[0], -1).T

train_x_f = flatten_func(train_x)
test_x_f = flatten_func(test_x)
print('train_x_f dimension is', train_x_f.shape)
print('test_x_f dimension is', test_x_f.shape)

# Standard data
# if the data is the picture, just divede 255
train_x_s = train_x_f/255
test_x_s = test_x_f/255

# sigmoid
# sigmoid loss function:  -y*log(a)-(1-y)*log(1-a)
def sigmoid_func(z):
    return 1/(1+np.exp(-z))

# initialize the w, b >> all 0
def initialize_with_zeros(dim):
    # dim: the dimension of w
    w = np.zeros(shape=(dim, 1))
    b = 0
    return w, b

def propagate(w, b, X, Y):
    m = X.shape[1]

    # forward-propagation
    A = sigmoid_func(np.dot(w.T, X) + b)
    # the loss function of logistic
    cost = (-1/m)*np.sum(Y*np.log(A)+(1-Y)*(np.log(1-A)))
    cost = np.squeeze(cost)
    # back-propagation
    dw = (1/m)*(np.dot(X, (A-Y).T))
    db = (1/m)*np.sum(A-Y)

    # store dw&db
    grads = {'dw': dw, 'db': db}
    return grads, cost

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    # for optimizing the w, b
    costs = []
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        dw = grads['dw']
        db = grads['db']

        w = w - learning_rate*dw
        b = b - learning_rate*db

        #
        costs.append(cost)
        print('iteration is %i, the cost is %f' % (i, cost))

    params = {'w': w, 'b': b}
    grads = {'dw': dw, 'db': db}
    return params, grads, costs

def predict_func(w, b, X):
    m = X.shape[1]
    y_pre = np.zeros((1, m))
    A = sigmoid_func(np.dot(w.T, X) + b)
    for i in range(m):
        y_pre[0, i] = 1 if A[0, i] >= 0.5 else 0
    return y_pre



# w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1, 2], [3, 4]]), np.array([[1, 0]])
# params, grads, costs = optimize(w, b, X, Y, 100, 0.009)
# predict(w, b, X)
# built model
w, b = initialize_with_zeros(train_x_s.shape[0])

parameters, grads, costs = optimize(w, b, train_x_s, train_y, 1000, 0.05)
w = parameters['w']
b = parameters['b']
y_pre_train = predict_func(w, b, train_x_s)
y_pre_test = predict_func(w, b, test_x_s)


# accuracy
print('The accuracy in train_dataset is: ', 100 - np.mean(abs(train_y - y_pre_train))*100)
print('The accuracy in test_dataset is: ', 100 - np.mean(abs(test_y - y_pre_test))*100)

