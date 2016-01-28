import numpy as np
import math
import operator as op
from scipy.stats import norm
import sys
import copy
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

def get_unshuffle_mapping(to_unshuffle, order):
    hd = heapdict()
    for x in xrange(to_unshuffle.size):
        hd[x] = -to_unshuffle[x]
    items_ordered = []
    while hd.keys():
        items_ordered.append(hd.popitem())
    new_ordering = frac_ordering(order)
    unshuffle_mapping = []
    for idx, member in enumerate(new_ordering):
        unshuffle_mapping.append((items_ordered[idx][0], member))
    return unshuffle_mapping, items_ordered

def apply_unshuffle_mapping(unshuffle_mapping, scrambled_data):
    unscrambled_data = np.zeros_like(scrambled_data)
    for member in unshuffle_mapping:
        scrambled, unscrambled = member
        unscrambled_data[unscrambled] = scrambled_data[scrambled]
    return unscrambled_data

def frac_ordering(order):
    layers = [[0]]
    while order > 0:
        old_layers = copy.deepcopy(layers)
        order -= 1
        # this would be less of a mess with one-indexing
        curr = (layers[-1][-1] + 1) * 2 - 1
        layers.append([])
        prev_old = 0
        next_old = old_layers[0][0]
        for idx, old_layer in enumerate(old_layers):
            for member in old_layer:
                next_old = member
                old_diff = next_old - prev_old
                curr -= old_diff
                layers[-(idx + 1)].append(curr)
                prev_old = member
    total_ordering = []
    for layer in layers:
        total_ordering += sorted(layer) # should I add to that sort?
    return total_ordering

def test_frac_ordering():
    assert frac_ordering(0) == [0]
    assert frac_ordering(1) == [0, 1]
    assert frac_ordering(2) == [0, 1, 2, 3]
    assert frac_ordering(3) == [0, 1, 2, 4, 3, 5, 6, 7]
    # this is where the addition problem comes in
    assert frac_ordering(4) == [0, 1, 2, 4, 8, 3, 5, 6, 9, 10, 12, 7, 11, 13, 14, 15]
    print "good!"

def test_frac_unshuffling():
    data = kron_line(10)
    plt.plot(data)
    npr.shuffle(data)
    mapping, _ = get_unshuffle_mapping(data, 10)
    unscrambled_data = apply_unshuffle_mapping(mapping, data)
    plt.plot(unscrambled_data)
    plt.show()

def test_mlp_delta():
    mlp_delta = np.load("delta.npy")
    mlp_delta = np.abs(mlp_delta)
    mlp_delta /= np.max(mlp_delta)
    mlp_delta = mlp_delta[:2048]
    mapping, _ = get_unshuffle_mapping(mlp_delta, 11)
    unscrambled_delta = apply_unshuffle_mapping(mapping, mlp_delta)
    plt.plot(unscrambled_delta)
    plt.show()

def get_noise(n=2048):
    noise = []
    for k in xrange(n):
        noise.append(norm.rvs(scale=1))
    return np.array(noise)

def test_noise():
    noise = get_noise()
    noise = np.abs(noise)
    noise /= np.max(noise)
    mapping, _ = get_unshuffle_mapping(noise, 11)
    unscrambled_noise = apply_unshuffle_mapping(mapping, noise)
    plt.plot(unscrambled_noise)
    plt.show()

if __name__ == "__main__":
    test_frac_ordering()
