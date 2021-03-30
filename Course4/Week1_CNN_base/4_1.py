# -*- coding: utf-8 -*-
"""
@Time    : 2021/3/27 22:23
@Author  : yany
@File    : 4_1.py
"""

import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.python.framework import ops
import time
from utils.cnn_utils import *

np.random.seed(1)

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
index = 6
plt.imshow(X_train_orig[index])
print("y = " + str(np.squeeze(Y_train_orig[:, index])))




X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0],-1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0],-1).T

X_train = X_train_flatten / 255
X_test = X_test_flatten / 255

#onehot
Y_train = convert_to_one_hot(Y_train_orig,6)
Y_test = convert_to_one_hot(Y_test_orig,6)




def create_placeholders(n_x, n_y):
    """
    Create placeholders for TensorFlow sessions
    Parameters.
        n_x - a real number, the size of the image vector (64*64*3 = 12288)
        n_y - a real number, the number of categories (from 0 to 5, so n_y = 6)

    Returns
        X - a placeholder for the data input, dimension [n_x, None], dtype = "float"
        Y - a placeholder for the label corresponding to the input, of dimension [n_Y, None], dtype = "float"

    Tip.
        None is used because it gives us the flexibility to handle the number of samples provided by the placeholder. In fact, the number of samples during testing/training is different.

    """

    X = tf.placeholder(tf.float32, [n_x, None], name="X")
    Y = tf.placeholder(tf.float32, [n_y, None], name="Y")

    return X, Y




def initialize_parameters():
    """
    初始化神经网络的参数，参数的维度如下：
        W1 : [25, 12288]
        b1 : [25, 1]
        W2 : [12, 25]
        b2 : [12, 1]
        W3 : [6, 12]
        b3 : [6, 1]

    返回：
        parameters - 包含了W和b的字典


    """

    tf.set_random_seed(1)  # 指定随机种子

    W1 = tf.get_variable("W1", [25, 12288], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable("b1", [25, 1], initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2", [12, 25], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable("b2", [12, 1], initializer=tf.zeros_initializer())
    W3 = tf.get_variable("W3", [6, 12], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable("b3", [6, 1], initializer=tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}

    return parameters


tf.reset_default_graph()

with tf.Session() as sess:
    parameters = initialize_parameters()
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))


def forward_propagation(X, parameters):
    """
    Implementation of forward propagation of a model with the model structure LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX

    Parameters.
        X - placeholder for the input data, with dimensionality (number of input nodes, number of samples)
        parameters - dictionary with the parameters W and b

    Returns.
        Z3 - output of the last LINEAR node

    """

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    Z1 = tf.add(tf.matmul(W1, X), b1)  #
    # Z1 = tf.matmul(W1,X) + b1
    A1 = tf.nn.relu(Z1)  # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)  # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)  # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)  # Z3 = np.dot(W3,Z2) + b3

    return Z3



tf.reset_default_graph()
with tf.Session() as sess:
    X,Y = create_placeholders(12288,6)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X,parameters)
    print("Z3 = " + str(Z3))


def compute_cost(Z3, Y):
    """
    Calculation of costs

    Parameters.
        Z3 - the result of forward propagation
        Y - label, a placeholder, same dimension as Z3

    Returns.
        cost - cost value


    """
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    return cost

tf.reset_default_graph()

with tf.Session() as sess:
    X,Y = create_placeholders(12288,6)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X,parameters)
    cost = compute_cost(Z3,Y)
    print("cost = " + str(cost))







def model(X_train, Y_train, X_test, Y_test,
          learning_rate=0.0001, num_epochs=1500, minibatch_size=32,
          print_cost=True, is_plot=True):
    """
    Implementation of a three-layer TensorFlow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX

    Parameters.
        X_train - training set with dimension (input size (number of input nodes) = 12288, number of samples = 1080)
        Y_train - training set with number of classifications, dimension (output size (number of output nodes) = 6, number of samples = 1080)
        X_test - test set, dimensionality (input size (number of input nodes) = 12288, number of samples = 120)
        Y_test - number of classifications in the test set, dimensioned as (output size (number of output nodes) = 6, number of samples = 120)
        learning_rate - the learning rate
        num_epochs - the number of traversals of the entire training set
        mini_batch_size - the size of each mini-batch dataset
        print_cost - whether to print the cost, once every 100 generations
        is_plot - whether to plot the curve

    Returns.
        parameters - the parameters after learning


    """
    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []  #

    # placeholder
    X, Y = create_placeholders(n_x, n_y)

    parameters = initialize_parameters()

    Z3 = forward_propagation(X, parameters)

    cost = compute_cost(Z3, Y)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(num_epochs):

            epoch_cost = 0
            num_minibatches = int(m / minibatch_size)  #
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch

                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

                epoch_cost = epoch_cost + minibatch_cost / num_minibatches

            if epoch % 5 == 0:
                costs.append(epoch_cost)
                if print_cost and epoch % 100 == 0:
                    print("epoch = " + str(epoch) + "    epoch_cost = " + str(epoch_cost))

        if is_plot:
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title("Learning rate =" + str(learning_rate))
            plt.show()

        parameters = sess.run(parameters)

        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("训练集的准确率：", accuracy.eval({X: X_train, Y: Y_train}))
        print("测试集的准确率:", accuracy.eval({X: X_test, Y: Y_test}))

        return parameters



start_time = time.clock()
parameters = model(X_train, Y_train, X_test, Y_test)
end_time = time.clock()
print(str(end_time - start_time) + " s" )

theta = 15/50
theta*(1-theta)/50