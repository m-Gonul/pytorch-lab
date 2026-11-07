import numpy as np


def sqdist_one_loop(Xte, Xtr):
    D = np.empty((len(Xte), len(Xtr)))
    for i, x in enumerate(Xte):
        D[i] = ((Xtr - x) ** 2).sum(axis=1)       # (ntr, d) − (d,) → broadcasting
    return D


def sqdist_no_loop(Xte, Xtr):
    te = (Xte ** 2).sum(1)[:, None]               # (nte, 1)
    tr = (Xtr ** 2).sum(1)[None, :]               # (1, ntr)
    return np.maximum(te + tr - 2 * Xte @ Xtr.T, 0)


def knn_predict_np(D, ytr, k):
    idx = np.argsort(D, axis=1, kind="stable")[:, :k]   # (nte, k)
    C = int(ytr.max()) + 1
    votes = np.eye(C)[ytr[idx]].sum(axis=1)             # (nte, C)
    return votes.argmax(axis=1)


Xtr = np.array([[0, 0], [1, 3], [3, 1], [4, 4]], dtype=float)
ytr = np.array([0, 1, 0, 1])
D = sqdist_no_loop(np.array([[1.0, 2.0]]), Xtr)
print(D, knn_predict_np(D, ytr, 1), knn_predict_np(D, ytr, 3))
