# -*- coding: utf-8 -*-
"""
@Time    : 2021/3/9 16:29
@Author  : yany
@File    : 1_4.py
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from utils.testCases import *
from utils.dnn_utils import *
from utils.lr_utils import *


# set seed for get same results
np.random.seed(1)
def initialize_parameters(n_x, n_h, n_y):
    """
    like week 3
    two hidden
    """
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    parameters = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    return parameters

parameters = initialize_parameters(3, 2, 1)


def initialize_parameters_deep(layders_dims):
    """
    for initializing parameters, i.e. W1, b1
    W1: (layders_dim[1], )
    """
    np.random.seed(3)
    parameters = {}
    L = len(layders_dims)

    for l in range(1, L):
        parameters['W'+str(l)] = np.random.randn(layders_dims[l], layders_dims[l-1]) * 0.01
        parameters['b'+str(l)] = np.zeros((layders_dims[l], 1))

    return parameters
layders_dims = [5, 4, 3]
parameters = initialize_parameters_deep(layders_dims)


def linear_forward(A, W, b):
    """
    :param A: l-1
    :param W: l
    :param b: l
    """
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache

A, W, b = linear_forward_test_case()
Z, linear_cache = linear_forward(A, W, b)

def linear_activation_forward(A_pre, W, b, activation):
    """
    :param A: l - 1
    :param W: l
    :param b: l
    :param activation: 'sigmoid' or 'relu'
    :return:
    """
    Z, linear_cache = linear_forward(A_pre, W, b)
    if activation == 'sigmoid':
        A, activation_cache = sigmoid(Z)
    elif activation == 'relu':
        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)
    return A, cache

A_pre, W, b = linear_activation_forward_test_case()
A, linear_activation_cache = linear_activation_forward(A_pre, W, b, activation='sigmoid')
A, linear_activation_cache = linear_activation_forward(A_pre, W, b, activation='relu')


def L_model_forward(X, parameters):
    """
    :param X:
    :param parameters: initialize_parameters_deep
    :return:
    """
    caches = []
    A = X
    # because parameters have W & b, so //2
    L = len(parameters) // 2
    for l in range(1, L):
        A_pre = A
        A, cache = linear_activation_forward(A_pre, parameters['W'+str(l)], parameters['b'+str(l)], 'relu')
        caches.append(cache)
    AL, cache = linear_activation_forward(A, parameters['W'+str(L)], parameters['b'+str(L)], 'sigmoid')
    caches.append(cache)
    return AL, caches

X, parameters = L_model_forward_test_case()
AL, caches = L_model_forward(X, parameters)

def compute_cost(AL, Y):
    m = Y.shape[1]

    cost = -np.sum(np.multiply(np.log(AL), Y) + np.multiply(np.log(1-AL), (1-Y))) / m

    cost = np.squeeze(cost)
    return cost

Y, AL = compute_cost_test_case()
compute_cost(AL, Y)

def linear_backward(dZ, cache):
    A_pre, W, b = cache
    m = A_pre.shape[1]
    dW = np.dot(dZ, A_pre.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_pre = np.dot(W.T, dZ)
    return dA_pre, dW, db

dZ, linear_cache = linear_backward_test_case()
dA_pre, dW, db = linear_backward(dZ, linear_cache)


def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
    elif activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
    dA_pre, dW, db = linear_backward(dZ, linear_cache)
    return dA_pre, dW, db

AL, linear_activation_cache = linear_activation_backward_test_case()
dA_pre, dW, db = linear_activation_backward(AL, linear_activation_cache, 'sigmoid')
dA_pre, dW, db = linear_activation_backward(AL, linear_activation_cache, 'relu')


def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    dAL = -(np.divide(Y, AL) - np.divide(1-Y, 1-AL))

    current_cache = caches[L-1]
    grads['dA'+str(L)], grads['dW'+str(L)], grads['db'+str(L)] = linear_activation_backward(dAL, current_cache, 'sigmoid')

    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_pre_temp, dW_temp, db_temp = linear_activation_backward(grads['dA'+str(l+2)], current_cache, 'relu')
        grads['dA'+str(l + 1)] = dA_pre_temp
        grads['dW'+str(l + 1)] = dW_temp
        grads['db'+str(l + 1)] = db_temp
    return grads

AL, Y_assess, caches = L_model_backward_test_case()
grads = L_model_backward(AL, Y_assess, caches)


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        parameters['W'+str(l+1)] = parameters['W'+str(l+1)] - learning_rate * grads['dW'+str(l+1)]
        parameters['b'+str(l+1)] = parameters['b'+str(l+1)] - learning_rate * grads['db'+str(l+1)]
    return parameters

parameters, grads = update_parameters_test_case()
parameters = update_parameters(parameters, grads, 0.1)


def two_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False, isPlot=True):

    np.random.seed(1)
    grads = {}
    costs = []
    (n_x, n_h, n_y) = layers_dims

    # initialize
    parameters = initialize_parameters(n_x, n_h, n_y)

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # forward
    for i in range(0, num_iterations):
        # 1
        A1, cache1 = linear_activation_forward(X, W1, b1, "relu")
        # 2
        A2, cache2 = linear_activation_forward(A1, W2, b2, "sigmoid")

        cost = compute_cost(A2, Y)

        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))

        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, "sigmoid")
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, "relu")

        grads["dW1"] = dW1
        grads["db1"] = db1
        grads["dW2"] = dW2
        grads["db2"] = db2

        parameters = update_parameters(parameters, grads, learning_rate)
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print(i, ", cost: ", np.squeeze(cost))
    if isPlot:
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

    return parameters


def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False, isPlot=True):
    np.random.seed(1)
    costs = []

    parameters = initialize_parameters_deep(layers_dims)
    print(parameters.keys())
    for i in range(0, num_iterations):
        AL, caches = L_model_forward(X, parameters)

        cost = compute_cost(AL, Y)

        grads = L_model_backward(AL, Y, caches)

        parameters = update_parameters(parameters, grads, learning_rate)

        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print(i, "cost：", np.squeeze(cost))
    if isPlot:
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
    return parameters


def predict(X, y, parameters):

    m = X.shape[1]
    n = len(parameters) // 2
    p = np.zeros((1, m))

    probas, caches = L_model_forward(X, parameters)

    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    print("accuracy: " + str(float(np.sum((p == y)) / m)))

    return p

