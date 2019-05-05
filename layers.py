from abc import ABC, abstractmethod
import functional

import numpy as np

class Layer(ABC):
    def __init__(self, in_size, out_size, activation_f):

        self.in_size = in_size
        self.out_size = out_size

        self.activation = activation_f

    @abstractmethod
    def feed(self, X):
        pass


class Dense(Layer):
    def __init__(self, in_size, out_size, activation_f=functional.sigmoid):

        super().__init__(in_size, out_size, activation_f)

        self.x = np.zeros((in_size, 1))
        self.W = np.random.rand(out_size, in_size) * 2 - 1
        self.W_grad = np.zeros((out_size, in_size))
        self.b = np.zeros((out_size, 1))
        self.b_grad = np.zeros((out_size, 1))
        self.z = np.zeros((out_size, 1))
        self.a = np.zeros((out_size, 1))

    def feed(self, x):
        self.x = x
        self.z = np.dot(self.W, self.x) + self.b
        self.a = self.activation(self.z)
        return self.a
