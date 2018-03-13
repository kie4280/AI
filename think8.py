import numpy as np
import _pickle as pickle
import os.path
import scipy.optimize as opt
import matplotlib.pyplot as plt

class Network:
    # parameters
    layer_num = 1
    node_num = 2
    load = True
    j = []
    data = np.array
    target = np.array

    # temp parameters
    synapses = []
    divide = 0
    output = None

    def __init__(self, hlayers=1, nodes=2, load=True):
        self.load = load
        if hlayers > 1:
            self.layer_num = hlayers
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

    def foward(self, data):
        layers = [data]

        for layer in range(self.layer_num + 1):
            layers.append(self.nonlin(np.dot(layers[layer], self.synapses[layer]), deriv=False))

        self.output = layers[-1]
        return layers

    def computecost(self, data, target):
        layers = self.foward(data)
        totalerror = 0.5 * sum((layers[-1] - target) ** 2)
        return totalerror

    def computegradient(self, data, target):
        layers = self.foward(data)
        layers.reverse()
        self.synapses.reverse()
        error = layers[0] - target
        deriv = []
        for layer in range(self.layer_num + 1):
            delta = error * self.nonlin(layers[layer], deriv=True)
            deriv.append((layers[layer + 1].T.dot(delta)).ravel())
            error = delta.dot(self.synapses[layer].T)
            # gradiant descent
        self.synapses.reverse()
        deriv.reverse()
        deriv = np.concatenate(deriv)

        return deriv

    def updateweights(self, params):
        start = 0
        for layer in range(self.layer_num + 1):
            x = self.synapses[layer].shape[0]
            y = self.synapses[layer].shape[1]
            self.synapses[layer] = np.reshape(params[start:start + x * y], (x, y))
            start += x * y

    def getweights(self):
        array = []
        for layer in range(self.layer_num + 1):
            array.append(self.synapses[layer].ravel())
        ravel = np.concatenate(array)
        return ravel

    def callbackf(self, params):
        self.updateweights(params)
        error = self.computecost(self.data, self.target)
        self.j.append(error)

    def funcwrapper(self, params, data, target):
        self.updateweights(params)
        cost = self.computecost(data, target)
        self.error = cost
        grad = self.computegradient(data, target)
        return cost, grad

    def train(self, data, target):

        self.data = data
        self.target = target
        if os.path.isfile("data.pkl") and self.load:
            # if False:
            self.synapses = self.read()
        else:
            # seed random numbers to make calculation
            # deterministic (just a good practice)
            np.random.seed(1)
            # initialize weights randomly with mean 0
            self.synapses.append(np.random.randn(data.shape[-1], self.node_num))
            for _ in range(self.layer_num - 1):
                self.synapses.append(np.random.randn(self.node_num, self.node_num))
            self.synapses.append(np.random.randn(self.node_num, target.shape[-1]))

        # scale the input and output data
        data_max = np.amax(self.data, axis=0)
        target_max = np.amax(self.target, axis=0)
        # X = np.true_divide(X, X_max)
        # Y = np.true_divide(Y, Y_max)
        self.data = self.data / data_max
        self.target = self.target / target_max

        # for iterator in range(1000):
        # out = self.computeDriv(data, target)
        params0 = self.getweights()

        options = {'maxiter': 200, 'disp': True}
        out = opt.minimize(fun=self.funcwrapper, x0=params0, args=(self.data, self.target), method='BFGS', options=options,
                           jac=True, callback=self.callbackf)
        self.updateweights(out.x)
        self.output = self.foward(data)[-1] * target_max
        a = np.arange(0, len(self.j), 1)
        plt.plot(a, self.j)
        plt.show()


# input dataset
X = np.array([[3, 5],
              [5, 1],
              [10, 2]
              ], dtype=float)

# output dataset
Y = np.array([[75, 82, 93]], dtype=float).T
network = Network(1, 3, False)
network.train(X, Y)
# network.save()
print("Output After Training:")
print(network.output)
