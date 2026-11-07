import numpy as np
from scipy import signal


def n_step(T, p0, n):
    return np.linalg.matrix_power(T, n) @ p0


def stationary(T):
    w, V = np.linalg.eig(T)
    v = np.real(V[:, np.argmin(np.abs(w - 1))])
    return v / v.sum()


def tv_curve(T, p0, n):
    w, V = np.linalg.eig(T)
    c = np.linalg.solve(V, p0)
    t = np.arange(n + 1)[:, None]
    P = np.real((V[None] * (w[None] ** t)[:, None, :] * c[None, None, :]).sum(-1))
    return 0.5 * np.abs(P - stationary(T)).sum(1)


def ar1_paths(phi, sigma, n, paths, seed=0):
    w = np.random.default_rng(seed).normal(0, sigma, (paths, n))
    w[:, 0] = 0
    return signal.lfilter([1.0], [1.0, -phi], w, axis=1)


def ensemble_stats(X):
    return X.mean(0), X.var(0)


T = np.array([[0.9, 0.5], [0.1, 0.5]])
print(n_step(T, np.array([1.0, 0]), 2), stationary(T))
