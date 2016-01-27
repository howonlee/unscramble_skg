import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

def kron_line(order):
    generator = np.array([1.0, 0.5])
    arr = generator.copy()
    for x in xrange(order-1):
        arr = np.kron(arr, generator)
    return arr

if __name__ == "__main__":
    plt.plot(kron_line(10))
    plt.show()
