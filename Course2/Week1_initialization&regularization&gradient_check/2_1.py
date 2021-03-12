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
from utils.init_utils import *
from utils.initialization import *

train_X, train_Y, test_X, test_Y = load_dataset(is_plot=True)


def model(X, Y, learning_rate=0.01, num_iterations=15000, print_cost=True, initialization="he", is_polt=True):
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

