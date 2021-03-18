# -*- coding: utf-8 -*-
"""
@Time    : 2021/3/17 16:16
@Author  : yany
@File    : exponentially_weighted_averages.py
"""
from utils.testCase import *


def initialize_velocity(parameters):
    """
    Initializing velocity, velocity is a dictionary of.
        - keys: "dW1", "db1", ... , "dWL", "dbL"
        - values:A matrix of values with the same value of zero as the corresponding gradient/parameter dimension.
    parameters.
        parameters - A dictionary containing the following parameters.
            parameters["W" + str(l)] = Wl
            parameters["b" + str(l)] = bl
    Returns:
        v - a dictionary variable containing the following parameters.
            v["dW" + str(l)] = the speed of dWl
            v["db" + str(l)] = speed of dbl

    """
    # the layers of the neural network
    L = len(parameters) // 2
    v = {}

    for l in range(L):
        v["dW" + str(l + 1)] = np.zeros_like(parameters["W" + str(l + 1)])
        v["db" + str(l + 1)] = np.zeros_like(parameters["b" + str(l + 1)])

    return v


# test initialize_velocity
print("-------------test initialize_velocity-------------")
parameters = initialize_velocity_test_case()
v = initialize_velocity(parameters)

print('v["dW1"] = ' + str(v["dW1"]))
print('v["db1"] = ' + str(v["db1"]))
print('v["dW2"] = ' + str(v["dW2"]))
print('v["db2"] = ' + str(v["db2"]))


def update_parameters_with_momentun(parameters, grads, v, beta, learning_rate):
    """
    Update parameters using momentum
    Parameters.
        parameters - A variable of type dictionary containing the following fields.
            parameters["W" + str(l)] = Wl
            parameters["b" + str(l)] = bl
        grads - a dictionary variable containing gradient values with the following fields.
            grads["dW" + str(l)] = dWl
            grads["db" + str(l)] = dbl
        v - dictionary variable containing the current speed, with the following fields.
            v["dW" + str(l)] = ...
            v["db" + str(l)] = ...
        beta - hyperparameter, momentum, real number
        learning_rate - learning rate, real number
    Returns.
        parameters - updated dictionary of parameters
        v - contains the updated velocity variables
    """
    L = len(parameters) // 2
    for l in range(L):
        # cal speed
        v["dW" + str(l + 1)] = beta * v["dW" + str(l + 1)] + (1 - beta) * grads["dW" + str(l + 1)]
        v["db" + str(l + 1)] = beta * v["db" + str(l + 1)] + (1 - beta) * grads["db" + str(l + 1)]

        # update parameters
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * v["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * v["db" + str(l + 1)]

    return parameters, v


# test update_parameters_with_momentun
print("-------------test update_parameters_with_momentun-------------")
parameters, grads, v = update_parameters_with_momentum_test_case()
update_parameters_with_momentun(parameters, grads, v, beta=0.9, learning_rate=0.01)

print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
print('v["dW1"] = ' + str(v["dW1"]))
print('v["db1"] = ' + str(v["db1"]))
print('v["dW2"] = ' + str(v["dW2"]))
print('v["db2"] = ' + str(v["db2"]))


