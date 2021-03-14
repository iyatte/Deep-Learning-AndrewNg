# -*- coding: utf-8 -*-
"""
@Time    : 2021/3/10 21:59
@Author  : yany
@File    : 2.py
"""
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
# initialization
from utils.init_utils import *
# regularization
from utils.reg_utils import *
# gradient checking
from utils.gc_utils import *
from utils.initialization import *
from utils.regularization import *

train_X, train_Y, test_X, test_Y = load_dataset(is_plot=True)

# initialization
def ini_model(X, Y, learning_rate=0.01, num_iterations=15000, print_cost=True, initialization="he", is_polt=True):
    """
    three layereds：LINEAR ->RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    :param X: input x data, dim is (2, the number of data)
    :param Y: input y data(0/1), dim is (1, the number of data)
    :param learning_rate: learning_rate
    :param num_iterations: num_iterations
    :param print_cost: weather print the cost(print the cost every 1000 times)
    :param initialization: type of initialization('zeros', 'random', 'he')
    :param is_polt: whether to draw a gradient descent curve
    :return: the parameters
    """

    grads = {}
    costs = []
    m = X.shape[1]
    layers_dims = [X.shape[0], 10, 5, 1]

    # select the method of initialization
    if initialization == "zeros":
        parameters = initialize_parameters_zeros(layers_dims)
    elif initialization == "random":
        parameters = initialize_parameters_random(layers_dims)
    elif initialization == "he":
        parameters = initialize_parameters_he(layers_dims)
    else:
        print("错误的初始化参数！程序退出")
        exit

    # begin
    for i in range(0, num_iterations):
        # forward
        a3, cache = forward_propagation(X, parameters)

        # cost
        cost = compute_loss(a3, Y)

        # backward
        grads = backward_propagation(X, Y, cache)

        # update parametes
        parameters = update_parameters(parameters, grads, learning_rate)

        # save cost
        if i % 1000 == 0:
            costs.append(cost)
            if print_cost:
                print(str(i) + " cost is " + str(cost))

    # plot the curve
    if is_polt:
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

    return parameters


# regularization

train_X, train_Y, test_X, test_Y = load_2D_dataset(is_plot=True)


def reg_model(X, Y, learning_rate=0.3, num_iterations=30000, print_cost=True, is_plot=True, lambd=0, keep_prob=1):
    """
    Implementation of a three-layer neural network: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID

    Parameters.
        X - input data, dimension (2, number to train/test)
        Y - label, [0 (blue) | 1 (red)], dimension (1, corresponding to the label of the input data)
        learning_rate - learning_rate
        num_iterations - num_iterations
        print_cost - whether to print the cost value, once per 10,000 iterations, but every 1000 iterations a cost value is recorded
        is_polt - whether to plot the gradient descent curve
        lambd - regularized hyperparameters, real numbers
        keep_prob - the probability of randomly deleting a node
    Return
        parameters - the parameters after learning
    """
    grads = {}
    costs = []
    m = X.shape[1]
    layers_dims = [X.shape[0], 20, 3, 1]

    # initialize_parameters
    parameters = initialize_parameters(layers_dims)

    # begin
    for i in range(0, num_iterations):
        # forward
        ## weather using keep_prob
        if keep_prob == 1:
            ### del the data not randomly
            a3, cache = forward_propagation(X, parameters)
        elif keep_prob < 1:
            ### del the data randomly
            a3, cache = forward_propagation_with_dropout(X, parameters, keep_prob)
        else:
            print("keep_prob is wrong!!!!!")
            exit

        # cost
        ## weather using L2 norm
        if lambd == 0:
            ### no
            cost = compute_cost(a3, Y)
        else:
            ### yeah
            cost = compute_cost_with_regularization(a3, Y, parameters, lambd)

        # backward

        ## lambd and keep_prob
        if (lambd == 0 and keep_prob == 1):
            ### lambd-no , keep_prom-no
            grads = backward_propagation(X, Y, cache)
        elif lambd != 0:
            ### lambd-yes , keep_prom-no
            grads = backward_propagation_with_regularization(X, Y, cache, lambd)
        elif keep_prob < 1:
            ### lambd-no , keep_prom-yes
            grads = backward_propagation_with_dropout(X, Y, cache, keep_prob)

        # update_parameters
        parameters = update_parameters(parameters, grads, learning_rate)

        # cost
        if i % 1000 == 0:
            costs.append(cost)
            if (print_cost and i % 10000 == 0):
                print("第" + str(i) + "次迭代，成本值为：" + str(cost))

    if is_plot:
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iterations (x1,000)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

    return parameters


# gradient check


def forward_propagation_n(X, Y, parameters):
    """
    Implementing forward propagation in a graph (and calculating the cost).

    Parameters.
        X - training set of m examples
        Y - labels for m examples
        parameters - python dictionary with parameters "W1", "b1", "W2", "b2", "W3", "b3".
            W1 - weight matrix, dimension (5,4)
            b1 - bias vector, dimension (5,1)
            W2 - weight matrix, dimension (3,5)
            b2 - bias vector, dimension (3,1)
            W3 - weight matrix, dimension (1,3)
            b3 - bias vector, dimension (1,1)

    Returns
        cost - cost function (logistic)
    """
    m = X.shape[1]
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)

    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)

    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    # cost
    logprobs = np.multiply(-np.log(A3), Y) + np.multiply(-np.log(1 - A3), 1 - Y)
    cost = (1 / m) * np.sum(logprobs)

    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)

    return cost, cache


def backward_propagation_n(X, Y, cache):
    """
    Implement the back propagation shown in the diagram.

    Parameters.
        X - input data points (number of input nodes, 1)
        Y - label
        cache - cache output from forward_propagation_n()

    Returns.
        gradients - A dictionary containing the cost gradients associated with each parameter, activation and pre-activation variable.
    """
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y
    dW3 = 1. / m * np.dot(dZ3, A2.T)
    db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1. / m * np.dot(dZ2, A1.T)
    db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1. / m * np.dot(dZ1, X.T)
    db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,
                 "dA2": dA2, "dZ2": dZ2, "dW2": dW2, "db2": db2,
                 "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients


def gradient_check_n(parameters, gradients, X, Y, epsilon=1e-7):
    """
    Check that backward_propagation_n calculates the cost gradient of the forward_propagation_n output correctly

    Parameters.
        parameters - python dictionary containing the parameters "W1", "b1", "W2", "b2", "W3", "b3".
        The output of grad_output_propagation_n contains the cost gradients associated with the parameters.
        x - input data points with dimension (number of input nodes, 1)
        y - label
        epsilon - calculates a small offset of the input to compute the approximate gradient

    Returns.
        difference - difference between approximate gradient and backward propagation gradient
    """
    # Initialization parameters
    parameters_values, keys = dictionary_to_vector(parameters)
    grad = gradients_to_vector(gradients)
    num_parameters = parameters_values.shape[0]
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))

    # cal gradapprox
    for i in range(num_parameters):
        # cal  J_plus [i]. input：“parameters_values，epsilon”. output=“J_plus [i]”
        thetaplus = np.copy(parameters_values)  # Step 1
        thetaplus[i][0] = thetaplus[i][0] + epsilon  # Step 2
        J_plus[i], cache = forward_propagation_n(X, Y, vector_to_dictionary(thetaplus))

        # cal J_minus [i]. input:“parameters_values，epsilon”. output=“J_minus [i]”。
        thetaminus = np.copy(parameters_values)  # Step 1
        thetaminus[i][0] = thetaminus[i][0] - epsilon  # Step 2
        J_minus[i], cache = forward_propagation_n(X, Y, vector_to_dictionary(thetaminus))

        # cal gradapprox[i]
        gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)

    # Compare gradapprox and backward propagation gradients by calculating differences.
    numerator = np.linalg.norm(grad - gradapprox)  # Step 1'
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)  # Step 2'
    difference = numerator / denominator  # Step 3'

    if difference < 1e-7:
        print("gradient checking: OK!")
    else:
        print("gradient checking: out of the threshold!")

    return difference
