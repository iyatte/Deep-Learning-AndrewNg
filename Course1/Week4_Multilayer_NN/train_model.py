# -*- coding: utf-8 -*-
"""
@Time    : 2021/3/9 21:11
@Author  : yany
@File    : train_model.py
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from utils.testCases import *
from utils.dnn_utils import *
from utils.lr_utils import *
from base_model import *

# # two
# train_set_x_orig , train_set_y , test_set_x_orig , test_set_y , classes = load_dataset()
#
# train_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
# test_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
#
# train_x = train_x_flatten / 255
# train_y = train_set_y
# test_x = test_x_flatten / 255
# test_y = test_set_y
# n_x = 12288
# n_h = 7
# n_y = 1
# layers_dims = (n_x,n_h,n_y)
#
# parameters = two_layer_model(train_x, train_set_y, layers_dims=(n_x, n_h, n_y), num_iterations = 2500, print_cost=True,isPlot=True)
#
#
# predictions_train = predict(train_x, train_y, parameters)
# predictions_test = predict(test_x, test_y, parameters)

# l
train_set_x_orig , train_set_y , test_set_x_orig , test_set_y , classes = load_dataset()

train_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

train_x = train_x_flatten / 255
train_y = train_set_y
test_x = test_x_flatten / 255
test_y = test_set_y
layers_dims = [12288, 20, 7, 5, 1] #  5-layer model
parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations=2500, print_cost = True,isPlot=True)


pred_train = predict(train_x, train_y, parameters)
pred_test = predict(test_x, test_y, parameters)



# analyze the miss-pre values
def print_mislabeled_images(classes, X, y, p):

    a = p + y
    mislabeled_indices = np.asarray(np.where(a == 1))
    plt.rcParams['figure.figsize'] = (40.0, 40.0)  # set default size of plots
    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[1][i]

        plt.subplot(2, num_images, i + 1)
        plt.imshow(X[:, index].reshape(64, 64, 3), interpolation='nearest')
        plt.axis('off')
        plt.title(
            "Prediction: " + classes[int(p[0, index])].decode("utf-8") + " \n Class: " + classes[y[0, index]].decode(
                "utf-8"))


print_mislabeled_images(classes, test_x, test_y, pred_test)
