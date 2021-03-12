# -*- coding: utf-8 -*-
"""
@Time    : 2021/3/12 20:51
@Author  : yany
@File    : initialization.py
"""
import numpy as np

"""
Different initialisation methods may lead to ultimately different performance

Random initialisation helps to break symmetry, allowing units in different hidden layers to learn different parameters.

Initialisation should not be done with too large an initial value.

He initialisation with the ReLU activation function can often give good results.

In deep learning, this can lead to some overfitting problems if the dataset is not large enough. The result of overfitting is a high accuracy on the training set, but a severe drop in accuracy when new samples are encountered.

"""


def initialize_parameters_zeros(layers_dims):
    """
    Set all parameters of the model to 0

    Parameters.
        layers_dims - list, the number of layers of the model and the number of nodes corresponding to each layer
    Returns
        parameters - a dictionary containing all W and b
            W1 - weight matrix with dimension (layers_dims[1], layers_dims[0])
            b1 - bias vector, dimension (layers_dims[1], 1)
            ---
            WL - weight matrix, dimension (layers_dims[L], layers_dims[L - 1])
            bL - bias vector, dimension (layers_dims[L],1)
    """
    parameters = {}
    # the number of layers
    L = len(layers_dims)

    for l in range(1, L):
        parameters["W" + str(l)] = np.zeros((layers_dims[l], layers_dims[l - 1]))
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))

    return parameters


def initialize_parameters_random(layers_dims):
    """
    the parameter's dim is same as above
    """
    # set random seed
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * 10
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

    return parameters


def initialize_parameters_he(layers_dims):
    # using `* np.sqrt(2 / layers_dims[l - 1])` to set the 'he' method

    np.random.seed(3)
    parameters = {}
    L = len(layers_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(2 / layers_dims[l - 1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))


    return parameters


