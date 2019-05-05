from network import NN
from layers import Dense

import mnist

import numpy as np

x_train = mnist.train_images() / 255
y_train = mnist.train_labels()

x_test = mnist.test_images() / 255
y_test = mnist.test_labels()

nn = NN([
    Dense(28*28, 32),
    Dense(32, 10)
])


x = []
y = []
for im, label in zip(x_train[:], y_train[:]):
    x.append(im.flatten().reshape(28*28, 1))

    y.append(np.array([0 for i in range(10)]))
    y[-1][label-1] = 1
    y[-1] = y[-1].reshape((10, 1))


nn.fit(x_train=x, y_train=y, epochs=25)

x = []
y = []

for im, label in zip(x_test, y_test):
    x.append(im.flatten().reshape(28*28, 1))

    y.append(np.array([0 for i in range(10)]))
    y[-1][label-1] = 1
    y[-1] = y[-1].reshape((10, 1))

nn.test(x, y)
