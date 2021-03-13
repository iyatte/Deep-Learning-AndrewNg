# -*- coding: utf-8 -*-
"""
@Time    : 2021/3/13 13:51
@Author  : yany
@File    : regularization.py
"""
import numpy as np
from utils.init_utils import *
from utils.reg_utils import *

def forward_propagation_with_dropout(X, parameters, keep_prob=0.5):
    """
    Implement forward propagation with randomly discarded nodes.
    LINEAR -> RELU + DROPOUT -> LINEAR -> RELU + DROPOUT -> LINEAR -> SIGMOID.

    Parameters.
        X - input data set, dimension (2, number of examples)
        parameters - python dictionary with parameters "W1", "b1", "W2", "b2", "W3", "b3".
            W1 - weight matrix, dimension (20,2)
            b1 - bias vector, dimension (20,1)
            W2 - weight matrix, dimension (3,20)
            b2 - bias vector, dimension (3,1)
            W3 - weight matrix, dimension (1,3)
            b3 - bias vector, dimension (1,1)
        keep_prob - probability of random deletion, real number
    Returns.
        A3 - final activation value, dimension (1,1), output of forward propagation
        cache - a tuple that stores some of the values used to calculate back propagation
    """
    np.random.seed(1)

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)

    # step 1：initialization- D1 = np.random.rand(..., ...)
    D1 = np.random.rand(A1.shape[0], A1.shape[1])
    # step 2: Convert the value of D1 to 0 or 1 (using keep_prob as a threshold)
    D1 = D1 < keep_prob
    # step 3: Discard some nodes of A1 (change its value to 0 or False)
    A1 = A1 * D1
    # step 4: Scale the value of the node that is not discarded (not 0)
    A1 = A1 / keep_prob

    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)

    # the following is same as above
    D2 = np.random.rand(A2.shape[0], A2.shape[1])
    D2 = D2 < keep_prob
    A2 = A2 * D2
    A2 = A2 / keep_prob

    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)

    return A3, cache


def backward_propagation_with_dropout(X, Y, cache, keep_prob):
    """
    Implements backward propagation of our randomly deleted model.
    Parameters.
        X - input dataset with dimension (2, number of examples)
        Y - label, dimension is (number of output nodes, number of examples)
        cache - the cache output from forward_propagation_with_dropout()
        keep_prob - probability of random deletion, real number

    Returns.
        gradients - a dictionary of gradient values for each parameter, activation value and pre-activation variable
    """
    m = X.shape[1]
    (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y
    dW3 = (1 / m) * np.dot(dZ3, A2.T)
    db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)
    dA2 = np.dot(W3.T, dZ3)
    # Step 1: Use the same nodes during forward propagation and discard those that are off (since any number multiplied by 0 or False is 0 or False)
    dA2 = dA2 * D2
    # Step 2: Scale the value of the node that is not discarded (not 0)
    dA2 = dA2 / keep_prob

    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1. / m * np.dot(dZ2, A1.T)
    db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)

    dA1 = dA1 * D1
    dA1 = dA1 / keep_prob
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1. / m * np.dot(dZ1, X.T)
    db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3, "dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1,
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients


def compute_cost_with_regularization(A3, Y, parameters, lambd):
    """
    Cost of implementing the L2 regularisation of Equation 2

    Parameters.
        A3 - output result of forward propagation in dimension (number of output nodes, number of training/testing)
        Y - label vector, one-to-one with the data, of dimension (number of output nodes, number of training/testing)
        parameters - a dictionary containing the parameters learned by the model
    Returns
        cost - the value of the regularisation loss calculated using equation 2

    """
    m = Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]

    cross_entropy_cost = compute_cost(A3, Y)

    L2_regularization_cost = lambd * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3))) / (2 * m)

    cost = cross_entropy_cost + L2_regularization_cost

    return cost



def backward_propagation_with_regularization(X, Y, cache, lambd):
    """
    Implements backward propagation of the model to which we have added L2 regularisation.

    Parameters.
        X - input dataset, dimensioned as (number of input nodes, number inside the dataset)
        Y - label, dimensioned as (number of output nodes, number inside the dataset)
        cache - the cache output from forward_propagation()
        lambda - regularization hyperparameter, real number

    Returns.
        gradients - a dictionary containing gradients for each parameter, activation value and pre-activation value variables
    """

    m = X.shape[1]

    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y

    dW3 = (1 / m) * np.dot(dZ3, A2.T) + ((lambd * W3) / m)
    db3 = (1 / m) * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = (1 / m) * np.dot(dZ2, A1.T) + ((lambd * W2) / m)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = (1 / m) * np.dot(dZ1, X.T) + ((lambd * W1) / m)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3, "dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1,
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients

