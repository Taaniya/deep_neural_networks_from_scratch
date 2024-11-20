import numpy as np


def sigmoid(x):
    """
    Computes sigmoid function output for input x, where can be a scalar value or a vector or a matrix.
    Args:
        x: a scalar or numpy array

    Returns:
        s: sigmoid(x) of same shape as input x
    """

    s = x / (1 + np.exp(-x))
    return s

def sigmoid_gradient(x):
    """
    Computes gradient of the sigmoid function
    sigmoid_gradient is computated as = s(1-s),
    where s = value of sigmoid function

    Args:
        x: scalar or numpy array

    Returns:
        s_gradient: gradient of sigmoid(x) of same shape as input x
    """
    s = x / (1 + np.exp(-x)) # compute sigmoid
    s_gradient = s * (1-s)
    return s_gradient
