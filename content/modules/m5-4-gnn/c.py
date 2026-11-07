import numpy as np


def dense_norm_adj(edge_index, n):
    ...


def propagate(edge_index, X, n):
    ...


ei = np.array([[0, 1, 1, 2], [1, 0, 2, 1]])
X = np.array([[1.0], [2.0], [3.0]])
print(dense_norm_adj(ei, 3) @ X, propagate(ei, X, 3))
