Course 3-Week 1
=========

## 1 ML Strategy
What can we optimise a model in many ways when we get a deep NN model.
* Collect more data
* Collect more diverse training set
* Train algorithm longer with gradient descent
* Try Adam instead of gradient descent
* Try bigger network
* Try smaller network
* Try dropout
* Add L2 regularization
* Network architecture: Activation functions, hidden units

## 2 Orthogonalization
### 2.1 Fit training set well on cost function
We can select more complex NN model, or use Adam etc.
### 2.2 Fit dev set well on cost function
By using regularization to get more train data.
### 2.3 Fit test set well on cost function
By using more test data.
### 2.4 Performs well in real world
Can realize it by changing dev data.

## 3 Single number evaluation metric
usually use F1 Score, P is Precision and R is Recall
$$F1 = \frac{2*P*R}{P+R}$$

## 4 Size of the dev and test sets
When sample numble is less then 10K:
Train:Dev:Test = 6:2:2  
or Train:Test = 7:3
else:  
Train:Dev:Test = 98:1:1  
or Train:Test = 99:1

## 5 Why human-level performance
![1](1.png)
In the graph, the horizontal coordinate is training time and the vertical coordinate is accuracy. Machine learning models are trained to approach or even exceed human-level performance. Theoretically, no model can exceed it, and the bayes optimal error represents the best performance.\

In fact, human-level performance can be very good in some areas. For example, in areas such as image recognition and speech recognition, humans are very good at it. Therefore, there is a need and a lot of effort to make machine learning models perform closer to human-level performance.

## 6  Improving your model performance
Improving the performance of machine learning models requires addressing two main issues: avoidable bias(Under-fitting) and variance(Overfitting).  

As we have previously described, the difference between training error and human-level error reflects avoidable bias, and the difference between dev error and training error reflects variance.

Common approaches to solving avoidable bias (Under-fitting) include

* Train bigger model
* Train longer/better optimization algorithms: momentum, RMSprop, Adam
* NN architecture/hyperparameters search

Common solutions to variance (Overfitting) include 

* More data
* Regularization: L2, dropout, data augmentation
* NN architecture/hyperparameters search

