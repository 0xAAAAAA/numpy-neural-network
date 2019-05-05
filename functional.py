import numpy as np

"""
Activation functions
"""

def sigmoid(x, deriv=False):
    sigm = 1. / (1. + np.exp(-x))
    if deriv:
        return sigm * (1. - sigm)
    return sigm


def relu(x, deriv=False):
    if deriv:
        return np.where(x > 0, 1.0, 0.0)
    return np.maximum(0, x)



"""
Loss functions
"""

def square_error(y_hat, y, deriv=False):
    if deriv:
        return 2 * (y_hat - y)
    return (y_hat - y)**2
