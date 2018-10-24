# MNIST simple feed-forward neural network
# Number of iterations and number of pictures to process
# can be altered in /setting.ini

import numpy as np, os, sys, math, time
from PIL import Image

# Get current directory
curr_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

np.seterr(all='warn')

def main():
    # If no arguments
    if len(sys.argv) < 2:
        return

    # "/nn.py train"
    elif sys.argv[1] == "train":

        # Read settings.ini
        settings_l =  open(curr_dir+"\\settings.ini", "r").readlines()
        pics, iters = int(settings_l[0].split('=')[1]), int(settings_l[1].split('=')[1])

        # Load training MNIST data
        train_images, train_labels = mnist_train_images(pics), mnist_train_labels(pics)

        # Create an NN instance
        nn = Network(28*28, 10)

        # Load weights from "/weights" if exist
        nn.load_weights()

        # Begin iterating through the training data
        print("Training ...")
        data_zip = zip(train_images[0:], train_labels[0:])
        for i in range(iters):
            for image, label in data_zip:
                nn.feedforward(image)
                nn.backprop(label)
        print("Done.")

        # Save "trained" weights
        nn.save_weights()
        return

    # "/nn.py test"
    elif sys.argv[1] == "test":

        # Load test MNIST data
        test_images, test_labels = mnist_test_images(), mnist_test_labels()

        # Create an instance of NN
        nn = Network(28*28, 10)

        # Load weights from /weights
        nn.load_weights()

        error = 0
        total = 0

        # Start iterating through the test data and calculating accuracy
        for image, label in zip(test_images, test_labels):
            total += 1
            nn.feedforward(image)
            if np.argmax(nn.layers[-1]) != np.argmax(label):
                error += 1

        # Print out the results
        print(f"Accuracy: {(total - error) / total * 100}%")

    # "/nn.py predict path"
    elif sys.argv[1] == "predict":
        """Predicts a picture whose path is specified in the arguments"""

        nn = Network(28*28, 10)
        nn.load_weights()
        nn.feedforward(load_single_pic(sys.argv[2]))
        print(np.argmax(nn.layers[-1]))

    else:
        print("Invalid argument")
        return

# NN class
class Network:
    def __init__(self, *layers):

        # Array of zeroed layers
        self.layers = [np.zeros((i, 1)) for i in layers]

        # Array of weights
        self.weights = [np.random.rand(layers[i + 1], layers[i]) * 2 - 1 for i in range(len(self.layers) - 1)]

        # Array of biases
        #self.biases = [np.random.rand(layers[i + 1], 1) for i in range(len(self.layers) - 1)]

    def feedforward(self, x):

        # Set input layer to be x
        self.layers[0] = x

        # Feed the input through all layers
        for i in range(len(self.weights)):
            self.layers[i + 1] = sigmoid(np.dot(self.weights[i], self.layers[i]))

    def backprop(self, target):

        # Create array of deltas with one element which is the output delta
        deltas = [(self.layers[-1] - target) * sigmoid_prime(self.layers[-1])]

        # Fill the deltas array with hidden deltas
        for i in range(len(self.weights) - 1):
            delta_l = [self.weights[-1-i].T.dot(deltas[-1 - i]) * sigmoid_prime(self.layers[-2 - i])]
            deltas = delta_l + deltas

        # Use deltas to adjust weights now
        for i in range(len(deltas)):
            self.weights[-1 - i] -= deltas[-1 - i].dot(self.layers[-2 - i].T)

    # Saves all weights as .npy file into /weights
    def save_weights(self):
        for weight, i in zip(self.weights, range(len(self.weights))):
            np.save(curr_dir+f"\\weights\\weight_l{i}", weight)

    # Loads weights from /weights
    def load_weights(self):
        for i in range(len(self.weights)):
            try:
                weight = np.load(curr_dir+f"\\weights\\weight_l{i}.npy")
                if np.shape(weight) != np.shape(self.weights[i]):
                    print(f"[Warning] Shape of weight_l{i} doesn't match the network's shape of that weight.")
                    continue
                self.weights[i] = weight
            except FileNotFoundError:
                print(f"[Warning] Couldn't load weight_l{i}.npy")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# (Prime with respect to sigmoid of x)
def sigmoid_prime(x):
    return x*(1-x)

def mnist_train_images(num):
    f_images = open(curr_dir+"\\mnist\\train-images", "rb")
    f_images.seek(16)
    images = []
    for i in range(num):
        img = [ord(f_images.read(1))/255 for i in range(28*28)]
        images.append(np.asarray([img]).T)
    f_images.close()
    return images

def mnist_train_labels(num):
    f_labels = open(curr_dir+"\\mnist\\train-labels", "rb")
    f_labels.seek(8)
    labels = []
    for i in range(num):
        n = ord(f_labels.read(1))
        vec = [0 for i in range(10)]
        vec[n] = 1
        labels.append(np.asarray([vec]).T)
    f_labels.close()
    return labels

def mnist_test_images(num=10000):
    f_images = open(curr_dir+"\\mnist\\test-images", "rb")
    f_images.seek(16)
    images = []
    for i in range(num):
        img = [ord(f_images.read(1))/255 for i in range(28*28)]
        images.append(np.asarray([img]).T)
    f_images.close()
    return images

def mnist_test_labels(num=10000):
    f_labels = open(curr_dir+"\\mnist\\test-labels", "rb")
    f_labels.seek(8)
    labels = []
    for i in range(num):
        n = ord(f_labels.read(1))
        vec = [0 for i in range(10)]
        vec[n] = 1
        labels.append(np.asarray([vec]).T)
    f_labels.close()
    return labels

def load_single_pic(pic_name):
    image = Image.open(curr_dir+f"\\{pic_name}.jpg")
    bitmap = list(image.getdata())
    for i in range(28*28):
        bitmap[i] = (255 - bitmap[i][0])/255
    return np.asarray([bitmap]).T

if __name__ == "__main__":
    main()
