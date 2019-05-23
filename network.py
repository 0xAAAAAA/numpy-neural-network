import numpy as np
import functional

class Network:

    def __init__(self, layers=[]):
        self.layers = layers

    def fit(self, x_train, y_train, epochs=5, loss_f=functional.MSE, lr=0.001):
        for _ in range(epochs):

            for x, y in zip(x_train, y_train):

                self.feed(x)
                self.backprop(y, loss_f, lr)

                error = np.sum(loss_f(self.layers[-1].a, y))

            print('Epoch {}/{}. Error: {}'.format(_ + 1, epochs, error))

    def feed(self, x):
        for L in self.layers:
            x = L.feed(x)
        return x

    def compute_activation_grads(self, a):
        """
        Computes weight and bias gradients
        of a single output activation
        Args:
        a - index of target activation
        """

        prev_delta = None

        for l in reversed(range(len(self.layers))):

            # If it's the final layer
            if l + 1 == len(self.layers):
                delta = np.zeros_like(self.layers[l].a)
                delta[a, 0] = self.layers[l].activation(self.layers[l].z, deriv=True)[a, 0]
            else:
                p1 = np.dot(self.layers[l + 1].W.T, prev_delta)
                p2 = self.layers[l].activation(self.layers[l].z, deriv=True)
                delta = p1 * p2
            prev_delta = delta

            self.layers[l].W_grad = np.dot(delta, self.layers[l].x.T)
            self.layers[l].b_grad = delta

    def compute_loss_f_grads(self, y, loss_f):
        """
        Computes weight and bias gradients of a loss function
        """

        prev_delta = None

        for l in reversed(range(len(self.layers))):

            # If it's the final layer
            if l + 1 == len(self.layers):
                a_grad = loss_f(self.layers[l].a, y, deriv=True)
                delta = a_grad * self.layers[l].activation(self.layers[l].z, deriv=True)
            else:
                p1 = np.dot(self.layers[l + 1].W.T, prev_delta)
                p2 = self.layers[l].activation(self.layers[l].z, deriv=True)
                delta = p1 * p2
            prev_delta = delta

            # If it's the first hidden layer
            # NOTE: self.layers[l].x = self.layers[l - 1].a
            self.layers[l].W_grad = np.dot(delta, self.layers[l].x.T)
            self.layers[l].b_grad = delta

    def backprop(self, y, loss_f, lr=0.001):

        self.compute_loss_f_grads(y, loss_f)

        # Subtract computed gradients from weights and biases
        for l in range(len(self.layers)):
            self.layers[l].W -= (self.layers[l].W_grad * lr)
            self.layers[l].b -= (self.layers[l].b_grad * lr)
