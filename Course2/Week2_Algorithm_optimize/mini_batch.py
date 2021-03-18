# -*- coding: utf-8 -*-
"""
@Time    : 2021/3/17 16:14
@Author  : yany
@File    : mini_batch.py
"""
import math
from utils.testCase import *


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """
    Create a random mini-batch list from (X, Y)

    Parameters.
        X - input data with dimension (number of input nodes, number of samples)
        Y - corresponding to the label of X, [1 | 0] (blue|red), dimension (1,number of samples)
        mini_batch_size - the number of samples per mini-batch

    Returns.
        mini-bacthes - a synchronized list with dimension (mini_batch_X,mini_batch_Y)
    """
    # set random seed
    np.random.seed(seed)
    m = X.shape[1]
    mini_batches = []

    # Step 1: Break up the order
    # It returns a random array of length m and the numbers in it are 0 to m-1
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]  # 将每一列的数据按permutation的顺序来重新排列。
    shuffled_Y = Y[:, permutation].reshape((1, m))

    # Second step, split
    # How many parts to split your training set into, note that if the value is 99.99, then the return value is 99 and the remaining 0.99 will be discarded
    num_complete_minibatches = math.floor(m / mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # If the size of the training set is exactly an integer multiple of the mini_batch_size, then we're done here
    # If the size of the training set is not an integer multiple of mini_batch_size, then there must be some left over at the end and we have to process it
    if m % mini_batch_size != 0:
        # Get the last remainder
        mini_batch_X = shuffled_X[:, mini_batch_size * num_complete_minibatches:]
        mini_batch_Y = shuffled_Y[:, mini_batch_size * num_complete_minibatches:]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


# test random_mini_batches
print("-------------test random_mini_batches-------------")
X_assess,Y_assess,mini_batch_size = random_mini_batches_test_case()
mini_batches = random_mini_batches(X_assess, Y_assess, mini_batch_size)

print("No.1 mini_batch_X 's dim is：", mini_batches[0][0].shape)
print("No.1 mini_batch_Y 's dim is：", mini_batches[0][1].shape)
print("No.2 mini_batch_X 's dim is：", mini_batches[1][0].shape)
print("No.2 mini_batch_Y 's dim is：", mini_batches[1][1].shape)
print("No.3 mini_batch_X 's dim is：", mini_batches[2][0].shape)
print("No.3 mini_batch_Y 's dim is：", mini_batches[2][1].shape)

