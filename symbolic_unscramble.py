import itertools

def prod_count(product_list):
    seen = set()
    ct = 0
    for member in product_list:
        canonical_member = "".join(sorted(member))
        if canonical_member not in seen:
            seen.add(canonical_member)
            ct += 1
    return ct

if __name__ == "__main__":
    initiator = "ABC"
    for x in xrange(15):
        prod_list = list(itertools.product(initiator, repeat=x))
        print prod_count(prod_list)
