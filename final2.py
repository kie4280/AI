import numpy as np
import _pickle as pickle
import os.path
import matplotlib.pyplot as plt


class Network:
    # parameters
    hidden_layer_num = 1
    node_num = 2
    load = False
    j = []
    data = np.array
    target = np.array
    # temp parameters
    synapses = []
    divide = 0
    output = None

    alpha = 1

    def __init__(self, hlayers=1, nodes=2, load=False):
        self.load = load
        if hlayers > 1:
            self.hidden_layer_num = hlayers
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

    def forward(self, data):
        layers = [data]
        for layer in range(self.hidden_layer_num + 1):
            layers.append(self.nonlin(np.dot(layers[layer], self.synapses[layer]), deriv=False))
        return layers

    def computecost(self, data, target):
        layers = self.forward(data)
        totalerror = 0.5 * sum((layers[-1] - target) ** 2)
        return totalerror

    def updateweights(self, params):
        start = 0
        for layer in range(self.hidden_layer_num + 1):
            x = self.synapses[layer].shape[0]
            y = self.synapses[layer].shape[1]
            self.synapses[layer] = np.reshape(params[start:start + x * y], (x, y))
            start += x * y

    def getweights(self):
        array = []
        for layer in range(self.hidden_layer_num + 1):
            array.append(self.synapses[layer].ravel())
        ravel = np.concatenate(array)
        return ravel

    def callbackf(self, params):
        self.updateweights(params)
        error = self.computecost(self.data, self.target)
        self.j.append(error)

    def minimize(self, data, target):

        layers = self.forward(data)
        layers.reverse()
        self.synapses.reverse()
        error = target - layers[0]
        self.j.append(0.5 * sum(error ** 2))
        previous = np.zeros((self.synapses[0].shape[0], self.synapses[0].shape[1]))

        for layer in range(self.hidden_layer_num + 1):
            delta = error * self.nonlin(layers[layer], deriv=True)
            error = np.dot(delta, self.synapses[layer].T)
            this = layers[layer + 1].T.dot(delta) * self.alpha
            self.synapses[layer] += (this + previous)
            previous = layers[layer + 1].T.dot(delta) * self.alpha

        self.synapses.reverse()

    def predict(self, data):
        return self.forward(data)[-1] * self.divide  # input dataset

    def train(self, data, target):

        self.data = data
        self.target = target
        if os.path.isfile("data.pkl") and self.load:
            self.synapses = self.read()
        else:
            np.random.seed(1)
            self.synapses.append(np.random.randn(data.shape[-1], self.node_num))
            for _ in range(self.hidden_layer_num - 1):
                self.synapses.append(np.random.randn(self.node_num, self.node_num))
            self.synapses.append(np.random.randn(self.node_num, target.shape[-1]))

        # scale the input and output data
        data_max = np.amax(self.data)
        target_max = np.amax(self.target)
        self.data = self.data / data_max
        self.target = self.target / target_max

        # datab = np.array_split(self.data, 2)
        # targetb = np.array_split(self.target, 2)

        for part in range(1000):
            self.minimize(self.data, self.target)
            # self.j.append(self.computecost(self.data, self.target))

        self.output = self.forward(self.data)[-1] * target_max
        x = range(len(self.j))
        plt.plot(x, self.j)
        plt.show()


X = np.array([[1, 7],
              [2, 8],
              [3, 9],
              [4, 10]
              ], dtype=float)

# target dataset
Y = np.array([[1, 2, 3, 4]], dtype=float).T
network = Network(1, 2, False)
network.train(X, Y)
