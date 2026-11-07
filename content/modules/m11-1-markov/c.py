import numpy as np
from scipy import signal


def n_step(T, p0, n):
    ...


def stationary(T):
    ...


def tv_curve(T, p0, n):
    ...


def ar1_paths(phi, sigma, n, paths, seed=0):
    ...


def ensemble_stats(X):
    ...


T = np.array([[0.9, 0.5], [0.1, 0.5]])
print(n_step(T, np.array([1.0, 0]), 2), stationary(T))
