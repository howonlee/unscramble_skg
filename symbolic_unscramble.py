import itertools
import collections

def prod_count(product_list):
    seen = set()
    ct = 0
    for member in product_list:
        canonical_member = "".join(sorted(member))
        if canonical_member not in seen:
            seen.add(canonical_member)
            ct += 1
    return ct

def prod_vec(product_list):
    canonicalized_list = ["".join(sorted(member)) for member in product_list]
    seen = collections.Counter(canonicalized_list)
    return seen.most_common()

if __name__ == "__main__":
    initiator = "ABCD"
    for x in xrange(5):
        prod_list = list(itertools.product(initiator, repeat=x))
        print prod_vec(prod_list)
