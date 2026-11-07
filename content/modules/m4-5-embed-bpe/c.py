import numpy as np


def embed_backward(ids, dout, V):
    ...


def pair_counts_np(ids, V):
    ...


def merge_ids(ids, a, b, new):
    ...


print(embed_backward(np.array([2, 0, 2]), np.array([[1.0, 1], [2, 2], [3, 3]]), 4))
print(merge_ids(np.array([0, 1, 2, 0, 1, 1, 0, 1]), 0, 1, 9))   # [9, 2, 9, 1, 9]
