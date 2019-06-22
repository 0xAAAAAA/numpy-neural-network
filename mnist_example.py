import numpy as np

from network import Network
from layers import Dense
from functional import CrossEntropy, relu, softmax

import mnist

x_train, y_train = mnist.train_images(), mnist.train_labels()
x_test, y_test = mnist.test_images(), mnist.test_labels()

# Flatten images
x_train = np.array([x.flatten().reshape((28*28, 1)) for x in x_train])
x_test = np.array([x.flatten().reshape((28*28, 1)) for x in x_test])

# Normalize images
x_train = x_train / 255.0
x_test = x_test / 255.0

# Convert labels to vectors
y_train = np.array([np.array([float(i == x)
                            for i in range(10)]).reshape((10, 1))
                                for x in y_train])

y_test = np.array([np.array([float(i == x)
                            for i in range(10)]).reshape((10, 1))
                                for x in y_test])

sample_size = 10000
x_train = x_train[:sample_size]
y_train = y_train[:sample_size]

nn = Network([
    Dense(28*28, 32, activation=relu),
    Dense(32, 10, activation=softmax)
])

print("Training...")
nn.fit(x_train, y_train, epochs=5, loss_f=CrossEntropy, lr=0.001)

print("Testing...")
total = 0
success = 0
for x, y in zip(x_test, y_test):

    total += 1

    nn.feed(x)

    success += int(np.argmax(nn.layers[-1].a) == np.argmax(y))

print('Accuracy: {}'.format(success/(total+1)))
