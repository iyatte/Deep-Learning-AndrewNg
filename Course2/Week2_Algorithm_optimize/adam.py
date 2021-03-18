# -*- coding: utf-8 -*-
"""
@Time    : 2021/3/17 16:16
@Author  : yany
@File    : adam.py
"""
from utils.testCase import *


def initialize_adam(parameters):
    """
    Initialize v and s, which are both dictionary type variables, both containing the following fields.
        - keys: "dW1", "db1", ... , "dWL", "dbL"
        - values: numpy matrix with zero values in the same dimension as the corresponding gradient/parameter

    parameters.
        parameters - A dictionary variable containing the following parameters.
            parameters["W" + str(l)] = Wl
            parameters["b" + str(l)] = bl
    Returns
        v - exponentially weighted average containing the gradient, with the following fields.
            v["dW" + str(l)] = ...
            v["db" + str(l)] = ...
        s - exponentially weighted average containing the squared gradient, with the following fields.
            s["dW" + str(l)] = ...
            s["db" + str(l)] = ...

    """

    L = len(parameters) // 2
    v = {}
    s = {}

    for l in range(L):
        v["dW" + str(l + 1)] = np.zeros_like(parameters["W" + str(l + 1)])
        v["db" + str(l + 1)] = np.zeros_like(parameters["b" + str(l + 1)])

        s["dW" + str(l + 1)] = np.zeros_like(parameters["W" + str(l + 1)])
        s["db" + str(l + 1)] = np.zeros_like(parameters["b" + str(l + 1)])

    return (v, s)


# test initialize_adam
print("-------------test initialize_adam-------------")
parameters = initialize_adam_test_case()
v, s = initialize_adam(parameters)

print('v["dW1"] = ' + str(v["dW1"]))
print('v["db1"] = ' + str(v["db1"]))
print('v["dW2"] = ' + str(v["dW2"]))
print('v["db2"] = ' + str(v["db2"]))
print('s["dW1"] = ' + str(s["dW1"]))
print('s["db1"] = ' + str(s["db1"]))
print('s["dW2"] = ' + str(s["dW2"]))
print('s["db2"] = ' + str(s["db2"]))


def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Updating parameters with Adam

    Parameters.
        parameters - A dictionary containing the following fields.
            parameters['W' + str(l)] = Wl
            parameters['b' + str(l)] = bl
        grads - A dictionary containing gradient values with the following key values.
            grads['dW' + str(l)] = dWl
            grads['db' + str(l)] = dbl
        v - Adam's variable, the moving average of the first gradient, is a dictionary type variable
        s - Adam's variable, the moving average of the squared gradient, is a dictionary type variable
        t - the number of current iterations
        learning_rate - the learning rate
        beta1 - momentum, a hyperparameter, used in the first stage so that the Y value of the curve does not start at 0 (see the graph of the weather data)
        beta2 - a parameter of RMSprop, hyperparameter
        epsilon - prevents the division by zero operation (denominator is 0)

    Returns.
        parameters - updated parameters
        v - moving average of the first gradient, a dictionary type variable
        s - the moving average of the squared gradient, a dictionary variable
    """
    L = len(parameters) // 2
    # Deviation-corrected values
    v_corrected = {}
    # Deviation-corrected values
    s_corrected = {}

    for l in range(L):
        # The moving average of the gradient, input: "v , grads , beta1", output: " v "
        v["dW" + str(l + 1)] = beta1 * v["dW" + str(l + 1)] + (1 - beta1) * grads["dW" + str(l + 1)]
        v["db" + str(l + 1)] = beta1 * v["db" + str(l + 1)] + (1 - beta1) * grads["db" + str(l + 1)]

        # Calculate the bias-corrected estimate for the first stage, enter "v , beta1 , t" , output: "v_corrected"
        v_corrected["dW" + str(l + 1)] = v["dW" + str(l + 1)] / (1 - np.power(beta1, t))
        v_corrected["db" + str(l + 1)] = v["db" + str(l + 1)] / (1 - np.power(beta1, t))

        # Calculate the moving average of the squared gradient, input: "s, grads , beta2", output: "s"
        s["dW" + str(l + 1)] = beta2 * s["dW" + str(l + 1)] + (1 - beta2) * np.square(grads["dW" + str(l + 1)])
        s["db" + str(l + 1)] = beta2 * s["db" + str(l + 1)] + (1 - beta2) * np.square(grads["db" + str(l + 1)])

        # Calculate the bias-corrected estimate for the second stage, input: "s , beta2 , t", output: "s_corrected"
        s_corrected["dW" + str(l + 1)] = s["dW" + str(l + 1)] / (1 - np.power(beta2, t))
        s_corrected["db" + str(l + 1)] = s["db" + str(l + 1)] / (1 - np.power(beta2, t))

        # update parameters, enter: "parameters, learning_rate, V_corrected, s_corrected, epsilon"Output: "parameters".
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * (
                    v_corrected["dW" + str(l + 1)] / np.sqrt(s_corrected["dW" + str(l + 1)] + epsilon))
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * (
                    v_corrected["db" + str(l + 1)] / np.sqrt(s_corrected["db" + str(l + 1)] + epsilon))

    return (parameters, v, s)


# test update_with_parameters_with_adam
print("-------------test update_with_parameters_with_adam-------------")
parameters, grads, v, s = update_parameters_with_adam_test_case()
update_parameters_with_adam(parameters, grads, v, s, t=2)

print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
print('v["dW1"] = ' + str(v["dW1"]))
print('v["db1"] = ' + str(v["db1"]))
print('v["dW2"] = ' + str(v["dW2"]))
print('v["db2"] = ' + str(v["db2"]))
print('s["dW1"] = ' + str(s["dW1"]))
print('s["db1"] = ' + str(s["db1"]))
print('s["dW2"] = ' + str(s["dW2"]))
print('s["db2"] = ' + str(s["db2"]))



