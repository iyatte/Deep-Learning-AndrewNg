# -*- coding: utf-8 -*-
"""
@Time    : 2021/3/18 15:06
@Author  : yany
@File    : 2_2.py
"""

from utils.opt_utils import *
from adam import *
from exponentially_weighted_averages import *
from mini_batch import *


def update_parameters_with_gd(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent

    Parameters.
        parameters - Dictionary containing the parameters to be updated.
            parameters['W' + str(l)] = Wl
            parameters['b' + str(l)] = bl
        grads - a dictionary containing each gradient value to update the parameters
            grads['dW' + str(l)] = dWl
            grads['db' + str(l)] = dbl
        learning_rate - the learning rate

    Return values.
        parameters - dictionary with updated parameters
    """

    L = len(parameters) // 2

    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters


def model(X, Y, layers_dims, optimizer, learning_rate=0.0007,
          mini_batch_size=64, beta=0.9, beta1=0.9, beta2=0.999,
          epsilon=1e-8, num_epochs=10000, print_cost=True, is_plot=True):
    """
    A 3-layer neural network model that can be run in different optimiser modes.

    Parameters.
        X - input data with dimension (2, the number of samples inside the input dataset)
        Y - the label corresponding to X
        layers_dims - a list containing the number of layers and the number of nodes
        optimizer - a string type parameter to select the type of optimisation, ["gd" | "momentum" | "adam" ]
        learning_rate - the learning rate
        mini_batch_size - the size of each small batch data set
        beta - a hyperparameter for momentum optimization
        beta1 - a hyperparameter for calculating the estimate of the exponential decay after the gradient
        beta1 - hyperparameter for computing an estimate of the exponential decay after a squared gradient
        epsilon - a hyperparameter used to avoid the division by zero operation in Adam, generally unchanged
        num_epochs - the number of iterations of the entire training set, (video 2.9 Learning rate decay, at 1:55, called "generations" in the video), equivalent to the previous num_iteration
        print_cost - whether to print the error value, once per 1000 iterations of the dataset, but record an error value every 100 iterations, also known as once per 1000 generations
        is_plot - whether to plot the graph

    Returns.
        parameters - contains the parameters after learning

    """
    L = len(layers_dims)
    costs = []
    # Each minibatch studied increases by 1
    t = 0
    # set random seed
    seed = 10

    # initialize_parameters
    parameters = initialize_parameters(layers_dims)

    # select optimizer
    # Use gradient descent directly without using any optimizer
    if optimizer == "gd":
        pass
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)
    else:
        print("optimizer wrong!")
        exit(1)

    # start
    for i in range(num_epochs):
        # Define random minibatches, we add seeds after each traversal of the dataset to rearrange the dataset so that the order of the data is different each time
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)

        for minibatch in minibatches:
            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch

            # forward_propagation
            A3, cache = forward_propagation(minibatch_X, parameters)

            # compute_cost
            cost = compute_cost(A3, minibatch_Y)

            # backward_propagation
            grads = backward_propagation(minibatch_X, minibatch_Y, cache)

            # update parameters
            if optimizer == "gd":
                parameters = update_parameters_with_gd(parameters, grads, learning_rate)
            elif optimizer == "momentum":
                parameters, v = update_parameters_with_momentun(parameters, grads, v, beta, learning_rate)
            elif optimizer == "adam":
                t = t + 1
                parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, t, learning_rate, beta1, beta2,
                                                               epsilon)
        if i % 100 == 0:
            costs.append(cost)
            if print_cost and i % 1000 == 0:
                print("第" + str(i) + "次遍历整个数据集，当前误差值：" + str(cost))
    if is_plot:
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('epochs (per 100)')
        plt.title("Learning rate = " + str(learning_rate))
        plt.show()

    return parameters


train_X, train_Y = load_dataset(is_plot=True)


# gd
layers_dims = [train_X.shape[0],5,2,1]
parameters = model(train_X, train_Y, layers_dims, optimizer="gd",is_plot=True)

preditions = predict(train_X,train_Y,parameters)

plt.title("Model with Gradient Descent optimization")
axes = plt.gca()
axes.set_xlim([-1.5, 2.5])
axes.set_ylim([-1, 1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)


# momentum
layers_dims = [train_X.shape[0],5,2,1]
parameters = model(train_X, train_Y, layers_dims, beta=0.9, optimizer="momentum",is_plot=True)
preditions = predict(train_X,train_Y,parameters)

plt.title("Model with Momentum optimization")
axes = plt.gca()
axes.set_xlim([-1.5, 2.5])
axes.set_ylim([-1, 1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)


# adam
layers_dims = [train_X.shape[0], 5, 2, 1]
parameters = model(train_X, train_Y, layers_dims, optimizer="adam",is_plot=True)

preditions = predict(train_X, train_Y, parameters)

plt.title("Model with Adam optimization")
axes = plt.gca()
axes.set_xlim([-1.5, 2.5])
axes.set_ylim([-1, 1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)


