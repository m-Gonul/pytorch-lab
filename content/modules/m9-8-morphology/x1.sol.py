import numpy as np
from scipy import ndimage

CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)


def fill_holes(A):
    Ac = ~A
    X = np.zeros_like(A)
    X[0, :], X[-1, :], X[:, 0], X[:, -1] = Ac[0, :], Ac[-1, :], Ac[:, 0], Ac[:, -1]
    while True:
        Y = ndimage.binary_dilation(X, CROSS) & Ac
        if np.array_equal(Y, X):
            return ~X
        X = Y


def components(A):
    lab = np.zeros(A.shape, np.int32)
    n = 0
    while True:
        rest = A & (lab == 0)
        if not rest.any():
            return lab
        n += 1
        X = np.zeros_like(A)
        X[np.unravel_index(np.argmax(rest), A.shape)] = True
        while True:
            Y = ndimage.binary_dilation(X, np.ones((3, 3), bool)) & A
            if np.array_equal(Y, X):
                break
            X = Y
        lab[X] = n


A = np.zeros((7, 7), bool); A[1:6, 1:6] = True; A[3, 3] = False
print(fill_holes(A).sum())
