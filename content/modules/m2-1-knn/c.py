import numpy as np


def sqdist_one_loop(Xte, Xtr):
    ...


def sqdist_no_loop(Xte, Xtr):
    ...


def knn_predict_np(D, ytr, k):
    ...


Xtr = np.array([[0, 0], [1, 3], [3, 1], [4, 4]], dtype=float)
ytr = np.array([0, 1, 0, 1])
D = sqdist_no_loop(np.array([[1.0, 2.0]]), Xtr)
print(D, knn_predict_np(D, ytr, 1), knn_predict_np(D, ytr, 3))
