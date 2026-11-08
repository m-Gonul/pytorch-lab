import numpy as np


def minibatch_features(F, T):
    M = np.einsum("na,abc->nbc", F, T)
    d = np.abs(M[:, None] - M[None]).sum(-1)
    return np.exp(-d).sum(1)
