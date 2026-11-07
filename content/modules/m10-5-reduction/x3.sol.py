import numpy as np


def reduce_by_key(keys, vals):
    o = np.argsort(keys, kind="stable")
    k, v = np.asarray(keys)[o], np.asarray(vals)[o]
    starts = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
    return k[starts], np.add.reduceat(v, starts)


def segmented_max(vals, flags):
    return np.maximum.reduceat(np.asarray(vals), np.flatnonzero(flags))


print(reduce_by_key(np.array([3, 1, 3, 2, 1]), np.array([1.0, 2, 3, 4, 5])))
