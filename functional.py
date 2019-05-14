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

def softmax(x, deriv=False):
    if deriv:
        soft = softmax(x)
        return soft * (1 - soft)

    exps = np.exp(x - x.max())
    return exps / np.sum(exps)

"""
Loss functions
"""

# Mean Squared Error
def MSE(y_hat, y, deriv=False):
    if deriv:
        return 2 * (y_hat - y)
    return (y_hat - y)**2

def CrossEntropy(y_hat, y, deriv=False):

    y_is_0 = (y == 0)
    y_is_1 = (y == 1)

    C = y_hat.copy() * 0.0

    if deriv:
        C += (y_is_1 * (-1 / y_hat))
        C += (y_is_0 * (1 / (1 - y_hat)))
    else:
        C += (y_is_1 * -np.log(y_hat))
        C += (y_is_0 * -np.log(1 - y_hat))

    return C
