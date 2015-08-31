# CSE574-Handwritten-Digit-Classification-in-Python
Machine Learning Course at UB (Spring 2015)

In this assignment, your task is to implement a Multilayer Perceptron Neural Network and evaluate its
performance in classifying handwritten digits. After completing this assignment, you are able to understand:
 How Neural Network works and use Feed Forward, Back Propagation to implement Neural Network?
 How to setup a Machine Learning experiment on real data?
 How regularization plays a role in the bias-variance tradeo?

File included in this exercise:
1. mnist all.mat: Original dataset from MNIST. In this file, there are 10 matrices for testing set and 10
matrices for training set, which corresponding to 10 digits. You will have to split the training data
into training and validation data.
2. nnScript.py: Python script for this programming project. Contains function definitions -
{ preprocess(): performs some preprocess tasks, and output the preprocessed train, validation and
test data with their corresponding labels. You need to make changes to this function.
{ sigmoid(): compute sigmoid function. The input can be a scalar value, a vector or a matrix. You
need to make changes to this function.
{ nnObjFunction(): compute the error function of Neural Network. You need to make changes to
this function.
{ nnPredict(): predicts the label of data given the parameters of Neural Network. You need to make
changes to this function.
{ initializeWeights(): return the random weights for Neural Network given the number of unit in
the input layer and output layer.
