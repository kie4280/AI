import numpy as np

layers = 2


# sigmoid function
def nonlin(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


# input dataset
X = np.array([[3, 0],
              [1, 0],
              [2, 0],
              [0, 0]])

# output dataset
Y = np.array([[9, 1, 4, 0
               ]]).T

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

divid = max(np.amax(X), np.amax(Y))
X = X / divid
Y = Y / divid

# initialize weights randomly with mean 0
syn0 = 2 * np.random.random((2, 3)) - 1
syn1 = 2 * np.random.random((3, 3)) - 1
syn2 = 2 * np.random.random((3, 1)) - 1

for iterator in range(10000):
    # forward propagation
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1))
    l3 = nonlin(np.dot(l2, syn2))
    # print(str(l0.shape) + str(l1.shape) + str(l2.shape) + str(l3.shape))

    # how much did we miss?
    l3_error = Y - l3

    # if (iterator % 1000) == 0:
    #     print("Error:" + str(np.mean(np.abs(l3_error))))

    l3_delta = l3_error * nonlin(l3, deriv=True)
    l2_error = l3_delta.dot(syn2.T)
    l2_delta = l2_error * nonlin(l2, deriv=True)
    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error * nonlin(l1, deriv=True)

    syn0 += l0.T.dot(l1_delta)
    syn1 += l1.T.dot(l2_delta)
    syn2 += l2.T.dot(l3_delta)
l3 = l3 * divid

print("Output After Training:")
print(l3)
