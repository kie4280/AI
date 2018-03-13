import numpy as np
import _pickle as pickle
import os.path
import scipy.optimize as opt


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

    def compute(self, params, data, target):
        layers = [data]
        self.updateweights(params)
        for layer in range(self.layer_num + 1):
            layers.append(self.nonlin(np.dot(layers[layer], self.synapses[layer]), deriv=False))
        layers.reverse()
        self.synapses.reverse()
        error = target - layers[0]
        totalerror = 0.5 * sum(error ** 2)
        deriv = []
        for layer in range(self.layer_num + 1):
            delta = error * self.nonlin(layers[layer], deriv=True)
            deriv.append((layers[layer + 1].T.dot(delta)).ravel())
            error = delta.dot(self.synapses[layer].T)
        self.synapses.reverse()
        deriv.reverse()
        # for layer in range(self.layer_num):
        deriv = np.concatenate(deriv)
        self.output = layers[0]

        return totalerror, deriv

    def updateweights(self, params):
        start = 0
        for layer in range(self.layer_num - 1):
            x = self.synapses[layer].shape[0]
            y = self.synapses[layer].shape[1]
            self.synapses[layer] = np.reshape(params[start:start + x * y], (x, y))
            start += x * y
        # self.j.append(self.compute(self.getweights(), self.data, self.target))

    def getweights(self):
        array = []
        for layer in range(self.layer_num + 1):
            array.append(self.synapses[layer].ravel())
        ravel = np.concatenate(array)
        return ravel

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
            self.synapses.append(2 * np.random.random((data.shape[-1], self.node_num)) - 1)
            for _ in range(self.layer_num - 1):
                self.synapses.append(2 * np.random.random((self.node_num, self.node_num)) - 1)
            self.synapses.append(2 * np.random.random((self.node_num, target.shape[-1])) - 1)

        # scale the input and output data
        X_max = np.amax(data)
        Y_max = np.amax(target)
        self.divide = max(X_max, Y_max)
        # X = np.true_divide(X, X_max)
        # Y = np.true_divide(Y, Y_max)
        data = data / self.divide
        target = target / self.divide

        # for iterator in range(1000):
        # out = self.computeDriv(data, target)

        options = {'maxiter': 2000, 'disp': True}
        out = opt.minimize(fun=self.compute, x0=self.getweights(), args=(data, target), method='BFGS', options=options,
                           jac=True, callback=self.updateweights)
        self.compute(out.x, data, target)
        self.output = self.output * self.divide


# input dataset
X = np.array([[3, 5],
              [5, 1],
              [10, 2]
              ])

# output dataset
Y = np.array([[75, 82, 93]]).T
network = Network(3, 2, False)
network.train(X, Y)
# network.save()
print("Output After Training:")
print(network.output)
