import numpy as np


def baird_features():
    ...


def expected_td(w0, alpha, gamma, n_iter, mu):
    ...


def expected_tdc(w0, alpha, beta, gamma, n_iter, mu):
    ...


w0 = [1, 1, 1, 1, 1, 1, 10, 1]
print(expected_td(w0, 0.01, 0.99, 500, np.full(7, 1 / 7))[1][-1])
