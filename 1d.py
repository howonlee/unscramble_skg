import numpy as np
import math
import operator as op
import sys
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

done_already = set()

def frac_ordering(prev, total):
    if prev == -1:
        return -1
    done_already.add(prev)
    new_member = round(prev / 2.0)
    if new_member in done_already:
        new_member = round((total + prev) / 2.0)
        if new_member in done_already: #again
            return -1
    return new_member

def unshuffle(to_unshuffle):
    hd = heapdict()
    for x in xrange(to_unshuffle.size):
        hd[x] = -to_unshuffle[x]
    items_ordered = []
    while hd.keys():
        items_ordered.append(hd.popitem())
    return items_ordered

def test_frac_ordering(num):
    currs = []
    curr = 0
    for x in xrange(num):
        currs.append(curr)
        curr = frac_ordering(curr, num)
    return currs

if __name__ == "__main__":
    # 0, 1:7, 7:22
    print sorted(test_frac_ordering(64)[7:22])
    data = kron_line(6)
    print sorted(unshuffle(data)[7:22], key=op.itemgetter(0))
    # plt.plot(map(op.itemgetter(0), unshuffle(data)))
    plt.show()
    # print test_frac_ordering()[:22]
    # npr.shuffle(data)
    # # unshuffled_data = unshuffle(data)
    # # plt.plot(data)
    # # plt.plot(unshuffled_data)
    # # plt.show()
