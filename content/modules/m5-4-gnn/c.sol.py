import numpy as np


def dense_norm_adj(edge_index, n):
    A = np.eye(n)
    A[edge_index[0], edge_index[1]] = 1
    d = A.sum(1)
    return A / np.sqrt(d[:, None] * d[None, :])


def propagate(edge_index, X, n):
    loops = np.arange(n)
    src = np.concatenate([edge_index[0], loops])
    dst = np.concatenate([edge_index[1], loops])
    deg = np.bincount(dst, minlength=n)
    w = 1 / np.sqrt(deg[src] * deg[dst])
    out = np.zeros((n, X.shape[1]))
    np.add.at(out, dst, w[:, None] * X[src])
    return out


ei = np.array([[0, 1, 1, 2], [1, 0, 2, 1]])
X = np.array([[1.0], [2.0], [3.0]])
print(dense_norm_adj(ei, 3) @ X, propagate(ei, X, 3))
