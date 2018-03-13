import numpy as np
import _pickle as pickle
import os.path
import matplotlib.pyplot as plot


class Network:
    # parameters
    layer_num = 3
    node_num = 2
    load = True

    # temp parameters
    synapses = []
    layers = []
    divide = 0
    output = None

    def __init__(self, layers, nodes, load=True):
        self.load = load
        if layers > 3:
            self.layer_num = layers
        if nodes > 2:
            self.node_num = nodes

    # sigmoid function
    def nonlin(self, x, deriv=False):
        if deriv:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

    def read(self):
        indata = open("data.pkl", "rb")
        inarray = pickle.load(indata)
        indata.close()
        return inarray

    def save(self):
        outdata = open("data.pkl", "wb")
        pickle.dump(self.synapses, outdata)
        outdata.close()

    def train(self, data, example):

        if os.path.isfile("data.pkl") and self.load:
            # if False:
            self.synapses = self.read()
        else:
            # seed random numbers to make calculation
            # deterministic (just a good practice)
            np.random.seed(1)

            # initialize weights randomly with mean 0

            self.synapses.append(2 * np.random.random((data.shape[-1], self.node_num)) - 1)
            for _ in range(self.layer_num - 3):
                self.synapses.append(2 * np.random.random((self.node_num, self.node_num)) - 1)
            self.synapses.append(2 * np.random.random((self.node_num, example.shape[-1])) - 1)

        # scale the input and output data
        X_max = np.amax(data)
        Y_max = np.amax(example)
        # X = np.true_divide(X, X_max)
        # Y = np.true_divide(Y, Y_max)
        data = data / X_max
        example = example / Y_max
        cost = []
        num = []

        for iterator in range(10000):
            # forward propagation
            self.layers.clear()
            self.layers.append(data)
            for layer in range(self.layer_num - 1):
                self.layers.append(self.nonlin(np.dot(self.layers[layer], self.synapses[layer])))

            # print(str(l0.shape) + str(l1.shape) + str(l2.shape) + str(l3.shape))

            # how much did we miss?
            self.layers.reverse()
            self.synapses.reverse()
            error = example - self.layers[0]

            cost.append(sum(error * error) / 2)
            num.append(iterator)

            for layer in range(self.layer_num - 1):
                delta = error * self.nonlin(self.layers[layer], deriv=True)
                error = delta.dot(self.synapses[layer].T)
                self.synapses[layer] += self.layers[layer + 1].T.dot(delta)

            self.synapses.reverse()
            self.layers.reverse()
        self.output = self.layers[-1] * self.divide
        plot.plot(num, cost)
        plot.show()


# input dataset
X = np.array([[1, 7],
              [2, 8],
              [3, 9],
              [4, 10]
              ], dtype=float)

# target dataset
Y = np.array([[1, 2, 3, 4]], dtype=float).T
network = Network(6, 3, load=False)
network.train(X, Y)
network.save()
print("Output After Training:")
print(network.output)
