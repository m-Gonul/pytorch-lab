from itertools import product


def _tlu(w, b, x):
    return 1 if sum(wi * xi for wi, xi in zip(w, x)) + b >= 0 else 0


def truth_table_net(f, n):
    hw, hb = [], []
    for row in product((0, 1), repeat=n):
        if f(*row):
            hw.append([1 if r else -1 for r in row])
            hb.append(-(sum(row) - 0.5))
    ow = [1] * len(hw)
    return hw, hb, ow, -0.5


def evaluate(net, x):
    hw, hb, ow, ob = net
    h = [_tlu(w, b, x) for w, b in zip(hw, hb)]
    return _tlu(ow, ob, h)


xor = lambda a, b: a ^ b
print([evaluate(truth_table_net(xor, 2), x) for x in product((0, 1), repeat=2)])
