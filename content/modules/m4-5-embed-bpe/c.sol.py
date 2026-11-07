import numpy as np


def embed_backward(ids, dout, V):
    dE = np.zeros((V, dout.shape[1]))
    np.add.at(dE, ids, dout)
    return dE


def pair_counts_np(ids, V):
    return np.bincount(ids[:-1] * V + ids[1:], minlength=V * V).reshape(V, V)


def merge_ids(ids, a, b, new):
    m = (ids[:-1] == a) & (ids[1:] == b)
    out = ids.copy()
    out[:-1][m] = new
    keep = np.ones(len(ids), bool)
    keep[1:][m] = False
    return out[keep]


print(embed_backward(np.array([2, 0, 2]), np.array([[1.0, 1], [2, 2], [3, 3]]), 4))
print(merge_ids(np.array([0, 1, 2, 0, 1, 1, 0, 1]), 0, 1, 9))   # [9, 2, 9, 1, 9]
