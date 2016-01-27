import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from unscramble_1d import get_unshuffle_mapping,\
        apply_unshuffle_mapping,\
        frac_ordering

def rand_permutation(size):
    arr = np.identity(size)
    npr.shuffle(arr) # inplace, one axis only
    return arr

def total_shuffle(arr):
    new_arr = arr.copy()
    # inplace on the copy, on the one flattened axis
    npr.shuffle(new_arr.flat)
    return new_arr

def kron_net(order):
    # generator taken from the SKG paper
    generator = np.array([[0.9, 0.7], [0.7, 0.15]])
    arr = generator.copy()
    for x in xrange(order-1):
        arr = np.kron(arr, generator)
    return arr

def sample_net(arr):
    """
    Sample a network from a network ensemble
    """
    new_arr = np.zeros_like(arr)
    for x in xrange(arr.shape[0]):
        for y in xrange(arr.shape[1]):
            if npr.rand() < arr[x,y]:
                new_arr[x,y] = 1
    return new_arr

if __name__ == "__main__":
    kron_order = 10
    net = kron_net(kron_order)
    r_perm_mat = rand_permutation(2 ** kron_order)
    net = np.dot(net, r_perm_mat)
    summed_net = net.sum(axis=0)
    matching, _ = get_unshuffle_mapping(summed_net, kron_order)
    plt.plot(apply_unshuffle_matching(matching, summed_net))
    plt.show()
