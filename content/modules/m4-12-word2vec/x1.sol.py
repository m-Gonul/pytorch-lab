import numpy as np


def cooccurrence(pairs, V):
    C = np.zeros((V, V))
    np.add.at(C, (pairs[:, 0], pairs[:, 1]), 1)
    return C


def ppmi(C, shift=1.0):
    N = C.sum()
    with np.errstate(divide="ignore"):
        pmi = np.log(C * N / np.outer(C.sum(1), C.sum(0)))
    return np.maximum(0, pmi - np.log(shift))


def svd_embeddings(M, d):
    U, S, _ = np.linalg.svd(M)
    return U[:, :d] * np.sqrt(S[:d])
