import numpy as np
import _pickle as pickle
import os.path


class Network:
    # parameters
    layer_num = 3
    node_num = 2

    # input dataset
    X = np.array([[3, 0],
                  [1, 0],
                  [2, 0],
                  [0, 0]])

    # output dataset
    Y = np.array([[9, 1, 4, 0
                   ]]).T
    # temp parameters
    synapses = []
    layers = []
    divide = 0
    output = None

    def __init__(self, layers, nodes):
        if layers > 3:
            self.layer_num = layers
        if nodes > 2:
            self.node_num = nodes
        if os.path.isfile("data.pkl"):
        # if False:
            self.synapses = self.read()
        else:
            # seed random numbers to make calculation
            # deterministic (just a good practice)
            np.random.seed(1)

            # initialize weights randomly with mean 0

            self.synapses.append(2 * np.random.random((self.X.shape[-1], self.node_num)) - 1)
            for _ in range(self.layer_num - 3):
                self.synapses.append(2 * np.random.random((self.node_num, self.node_num)) - 1)
            self.synapses.append(2 * np.random.random((self.node_num, self.Y.shape[-1])) - 1)

        # scale the input and output data
        X_max = np.amax(self.X)
        Y_max = np.amax(self.Y)
        self.divide = max(X_max, Y_max)
        # X = np.true_divide(X, X_max)
        # Y = np.true_divide(Y, Y_max)
        self.X = self.X / self.divide
        self.Y = self.Y / self.divide

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

    def train(self):

        for iterator in range(10000):
            # forward propagation
            self.layers.clear()
            self.layers.append(self.X)
            for layer in range(self.layer_num - 1):
                self.layers.append(self.nonlin(np.dot(self.layers[layer], self.synapses[layer])))

            # print(str(l0.shape) + str(l1.shape) + str(l2.shape) + str(l3.shape))

            # how much did we miss?
            self.layers.reverse()
            self.synapses.reverse()
            errors = [self.Y - self.layers[0]]
            for layer in range(self.layer_num - 1):
                delta = errors[layer] * self.nonlin(self.layers[layer], deriv=True)
                errors.append(delta.dot(self.synapses[layer].T))
                self.synapses[layer] += self.layers[layer + 1].T.dot(delta)
            # l3_delta = l3_error * self.nonlin(l3, deriv=True)
            # l2_error = l3_delta.dot(self.synapses[2].T)
            # l2_delta = l2_error * self.nonlin(l2, deriv=True)
            # l1_error = l2_delta.dot(self.synapses[1].T)
            # l1_delta = l1_error * self.nonlin(l1, deriv=True)
            # if (iterator % 1000) == 0:
            # print("Error:" + str(np.mean(np.abs(l3_error))))
            # print(str(l0.shape) + str(l1.shape) + str(l2.shape) + str(l3.shape))
            # print(str(syn0.shape) + str(syn1.shape) + str(syn2.shape))
            # print(str(syn0.shape) + str(syn1.shape) + str(syn2.shape))
            # self.synapses[0] += l0.T.dot(l1_delta)
            # self.synapses[1] += l1.T.dot(l2_delta)
            # self.synapses[2] += l2.T.dot(l3_delta)
            # print("reverse")
            self.synapses.reverse()
            self.layers.reverse()
        self.output = self.layers[-1] * self.divide


network = Network(6, 3)
network.train()
network.save()
print("Output After Training:")
print(network.output)
