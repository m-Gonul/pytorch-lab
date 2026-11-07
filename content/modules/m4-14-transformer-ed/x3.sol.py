import numpy as np


def causal(T):
    return np.tril(np.ones((T, T), bool))


def sliding_window(T, w):
    i, j = np.indices((T, T))
    return (j <= i) & (i - j < w)


def prefix_lm(T, P):
    i, j = np.indices((T, T))
    return (j <= i) | ((i < P) & (j < P))


def strided(T, s):
    i, j = np.indices((T, T))
    return (j <= i) & ((i - j < s) | ((i - j) % s == 0))


def reach(allow, L):
    A = (allow | np.eye(len(allow), dtype=bool)).astype(int)
    R = np.eye(len(allow), dtype=int)
    for _ in range(L):
        R = np.minimum(R @ A, 1)
    return R.astype(bool)
