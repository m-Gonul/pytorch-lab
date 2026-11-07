import math
import numpy as np


def kahan_sum(x):
    s = np.float32(0)
    c = np.float32(0)
    for v in np.asarray(x, np.float32):
        y = v - c
        t = s + y
        c = (t - s) - y
        s = t
    return s


def _pairwise(x):
    x = np.asarray(x, np.float32)
    while len(x) > 1:
        if len(x) % 2:
            x = np.append(x, np.float32(0))
        x = x[0::2] + x[1::2]
    return x[0]


def errors(n, seed=0):
    x = np.random.default_rng(seed).random(n).astype(np.float32)
    ref = math.fsum(x.astype(np.float64))
    seq = np.cumsum(x, dtype=np.float32)[-1]
    rel = lambda v: abs(float(v) - ref) / ref
    return rel(seq), rel(_pairwise(x)), rel(kahan_sum(x)), rel(np.sum(x))


print(errors(100000))
