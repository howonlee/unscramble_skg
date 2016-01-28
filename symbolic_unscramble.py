import itertools
import collections
import numpy as np
import operator as op

def prod_count(product_list):
    seen = set()
    ct = 0
    for member in product_list:
        canonical_member = "".join(sorted(member))
        if canonical_member not in seen:
            seen.add(canonical_member)
            ct += 1
    return ct

def mult(member):
    prod = 1
    for digit in member:
        prod *= int(digit)
    return prod

def prod_vec(product_list):
    canonicalized_list = ["".join(sorted(member)) for member in product_list]
    canonicalized_list = [mult(member) for member in canonicalized_list]
    seen = collections.Counter(canonicalized_list)
    return seen.most_common()

if __name__ == "__main__":
    initiator = "7532"
    prod_list = list(itertools.product(initiator, repeat=2))
    mult_list = map(mult, prod_list)
    print np.argsort(-np.array(mult_list)).reshape(4,4)
    # for x in xrange(5):
    # sorted_vec = sorted(prod_vec(prod_list), key=op.itemgetter(0), reverse=True)
    # print [ls[1] for ls in sorted_vec]
