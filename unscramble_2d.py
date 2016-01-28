import numpy as np
import numpy.random as npr
import copy
import matplotlib.pyplot as plt

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
    generator = np.array([[1, 0.8], [0.3, 0.1]])
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

def shuffle_mat(net, kron_order):
    r_perm_mat = rand_permutation(2 ** kron_order)
    c_perm_mat = rand_permutation(2 ** kron_order)
    new_net = np.dot(net, r_perm_mat)
    new_net = np.dot(c_perm_mat, new_net)
    return new_net

def by_axis_unscrambling():
    """
    Can't work this way.
    """
    kron_order = 10
    net = kron_net(kron_order)
    net = shuffle_mat(net, kron_order)
    r_summed_net = net.sum(axis=0)
    r_matching, _ = get_unshuffle_mapping(r_summed_net, kron_order)
    r_unpermute_mat = np.zeros_like(net)
    for member in r_matching:
        r_unpermute_mat[member] = 1
    net = np.dot(net, r_unpermute_mat)
    c_summed_net = net.sum(axis=1)
    c_matching, _ = get_unshuffle_mapping(c_summed_net, kron_order)
    c_unpermute_mat = np.zeros_like(net)
    for member in c_matching:
        fst, snd = member
        c_unpermute_mat[snd, fst] = 1
    net = np.dot(c_unpermute_mat, net)
    plt.imshow(net)
    plt.show()

def get_unshuffle_mapping(stuff):
    pass

def apply_unshuffle_mapping(stuff):
    pass

def frac_ordering(order):
    generator = np.array([[7, 5], [3, 2]])
    arr = generator.copy()
    for x in xrange(order-1):
        arr = np.kron(arr, generator)
    return np.argsort(-arr, axis=None)

def test_frac_ordering():
    assert list(frac_ordering(1)) == [0, 1, 2, 3]
    assert list(frac_ordering(2)) == [0, 1, 2, 3, 4, 8, 6, 9, 5, 10, 7, 11, 12, 13, 14, 15]
    assert list(frac_ordering(3)) == [0, 1, 2, 4, 3, 5, 6, 16, 32, 8, 7, 10, 34, 12, 33, 17, 20, 9, 36, 18, 35, 14, 21, 22, 38, 37, 13, 11, 19, 40, 24, 48, 15, 39, 23, 42, 49, 28, 26, 41, 44, 50, 25, 52, 53, 30, 29, 43, 51, 46, 27, 45, 54, 56, 55, 31, 47, 57, 58, 60, 62, 59, 61, 63]
    print "happy times!"

if __name__ == "__main__":
    test_frac_ordering()
