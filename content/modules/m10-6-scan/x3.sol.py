import numpy as np


def counting_sort(keys, K):
    k = np.asarray(keys)
    hist = np.bincount(k, minlength=K)
    start = np.cumsum(hist) - hist
    onehot = (k[:, None] == np.arange(K)[None, :]).astype(np.int64)
    rank = (np.cumsum(onehot, 0) - onehot)[np.arange(len(k)), k]
    dest = start[k] + rank
    perm = np.empty(len(k), np.int64)
    perm[dest] = np.arange(len(k))
    return k[perm], perm


print(counting_sort(np.array([3, 1, 3, 0, 1, 2]), 4))
