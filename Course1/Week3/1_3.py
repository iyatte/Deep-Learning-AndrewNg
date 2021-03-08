# -*- coding: utf-8 -*-
"""
@Time    : 2021/3/8 13:58
@Author  : yany
@File    : 1_3.py
"""

import numpy as np
import matplotlib.pyplot as plt
from utils.testCases import *
from utils.planar_utils import *
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score

X, Y = load_planar_dataset()
print('X shape ', X.shape)
print('Y shape ', Y.shape)

plt.scatter(X[0, :], X[1, :], c=Y, cmap=plt.cm.Spectral)

# test the effect of logistic

clf = LogisticRegressionCV()
clf.fit(X.T, Y.T)

plot_decision_boundary(lambda x: clf.predict(x), X, Y)
lr_pre = clf.predict(X.T)
print('accuracy_score: ', accuracy_score(Y.T, lr_pre))


# Neural network
def layer_sizes(X, Y, n_h):
    n_x = X.shape[0]
    n_y = Y.shape[0]
    return n_x, n_h, n_y


def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(2)
    W1 = np.random.randn(n_h, n_x)*0.01
    b1 = np.zeros(shape=(n_h, 1))
    W2 = np.random.randn(n_y, n_h)*0.01
    b2 = np.zeros(shape=(n_y, 1))
    parameters = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    return parameters

n_x, n_h, n_y = layer_sizes(X, Y, 4)
parameters = initialize_parameters(n_x, n_h, n_y)


def forward_propafation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    Z1 = np.dot(W1, X) + b1
    # active function tanh
    A1 = np.tanh(Z1)

    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}
    return A2, cache

X_assess, parameters = forward_propagation_test_case()
A2, cache = forward_propafation(X_assess, parameters)
np.mean(cache['Z1'])
np.mean(cache['A1'])
np.mean(cache['Z2'])
np.mean(cache['A2'])


def compute_cost(A2, Y):
    m = Y.shape[1]
    # cost
    cost = float(-(1/m)*np.sum(np.multiply(Y, np.log(A2)) + np.multiply((1-Y), np.log(1-A2))))
    return cost


A2, Y_assess, parameters = compute_cost_test_case()

compute_cost(A2, Y_assess)


# backward
def backward_propagation(parameters, cache, X, Y):
    m = Y.shape[1]

    W1 = parameters['W1']
    W2 = parameters['W2']

    A1 = cache['A1']
    A2 = cache['A2']

    dZ2 = A2 - Y
    dW2 = (1/m) * np.dot(dZ2,  A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1-np.power(A1, 2))
    dW1 = (1/m) * np.dot(dZ1, X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
    return grads

parameters, cache, X_assess, Y_assess = backward_propagation_test_case()

grads = backward_propagation(parameters, cache, X_assess, Y_assess)


def update_parameters(parameters, grads, learning_rate=1.2):
    W1 = parameters['W1']
    W2 = parameters['W2']
    b1 = parameters['b1']
    b2 = parameters['b2']

    dW1 = grads['dW1']
    dW2 = grads['dW2']
    db1 = grads['db1']
    db2 = grads['db2']

    W1 = W1 - learning_rate * dW1
    W2 = W2 - learning_rate * dW2
    b1 = b1 - learning_rate * db1
    b2 = b2 - learning_rate * db2

    parameters = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    return parameters

parameters, grads = update_parameters_test_case()
parameters = update_parameters(parameters, grads)



# create neural model

def nn_model(X, Y, n_h, num_iterations, print_cost=False):
    np.random.seed(3)
    n_x, n_h, n_y = layer_sizes(X, Y, n_h)
    parameters = initialize_parameters(n_x, n_h, n_y)

    for i in range(num_iterations):
        A2, cache = forward_propafation(X, parameters)
        cost = compute_cost(A2, Y)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads, learning_rate=0.5)
        if print_cost:
            print(i, 'cost is', cost)
    return parameters

# test
X_assess, Y_assess = nn_model_test_case()
parameters = nn_model(X_assess, Y_assess, 4, 10000, False)

def nn_predict(parameters, X):
    A2, _ = forward_propafation(X, parameters)
    y_pre = np.round(A2)
    return y_pre



# begin


arameters = nn_model(X, Y, 4, 10000, True)

plot_decision_boundary(lambda x: nn_predict(arameters, X.T), X, Y)

y_pre = nn_predict(arameters, X)


