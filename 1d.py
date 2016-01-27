import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from heapdict import heapdict

def kron_line(order):
    generator = np.array([1.0, 0.5])
    arr = generator.copy()
    for x in xrange(order-1):
        arr = np.kron(arr, generator)
    return arr

def fractal_attractor():
    """
    Interesting phenomenon, but not unscrambling
    """
    data = kron_line(15)
    unshuffled_data = data.copy()
    npr.shuffle(data) # inplace
    for x in xrange(1000):
        npr.shuffle(data)
        data = np.abs(data - unshuffled_data)
    plt.plot(sorted(data))
    plt.show()

def frac_ordering(length):
############
############
############
############
    pass

def unshuffle(to_unshuffle):
    hd = heapdict()
    for x in xrange(to_unshuffle.size):
        hd[x] = -to_unshuffle[x]
    while hd.keys():
        print hd.popitem()

if __name__ == "__main__":
    data = kron_line(8)
    npr.shuffle(data)
    unshuffle(data)
    # unshuffled_data = unshuffle(data)
    # plt.plot(data)
    # plt.plot(unshuffled_data)
    # plt.show()
