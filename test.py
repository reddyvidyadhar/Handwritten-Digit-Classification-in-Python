import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt

def sigmoid(z):
     return  1.0/(1.0+np.exp(-z))
# No need to define new function to evaluate sigmoid of a matrix since the above sigmoid() function accommodates for both scalar and vector quantities
#def sigmoidmat(m):
#    return 1.0/(1.0+exp(-m))
