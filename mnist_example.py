from network import Network
from layers import *
from functional import *

import mnist

import numpy as np


def test(x_test, y_test, nn: Network):
    total = 0
    success = 0
    for x, y in zip(x_test, y_test):
        total += 1
        nn.feed(x)

        success += (np.argmax(nn.layers[-1].a) == np.argmax(y))

    print('Accuracy: {}'.format(success/total))

x_train = mnist.train_images() / 255
y_train = mnist.train_labels()

x_test = mnist.test_images() / 255
y_test = mnist.test_labels()

nn = Network([
    Dense(28*28, 32, activation=relu),
    Dense(32, 10, activation=softmax)
], CrossEntropy)


x = []
y = []
for im, label in zip(x_train[:], y_train[:]):
    x.append(im.flatten().reshape(28*28, 1))

    y.append(np.array([0 for i in range(10)]))
    y[-1][label-1] = 1
    y[-1] = y[-1].reshape((10, 1))


nn.fit(x_train=x, y_train=y, epochs=1)

x = []
y = []
for im, label in zip(x_test, y_test):
    x.append(im.flatten().reshape(28*28, 1))

    y.append(np.array([0 for i in range(10)]))
    y[-1][label-1] = 1
    y[-1] = y[-1].reshape((10, 1))

test(x, y, nn)
