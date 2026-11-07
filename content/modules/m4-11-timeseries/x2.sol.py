import numpy as np


def ensemble_stats(X):
    m = X.mean(0)
    Z = X - m
    return m, Z.T @ Z / len(X)


def wss_gap(X):
    m, K = ensemble_stats(X)
    T = K.shape[0]
    d = np.subtract.outer(np.arange(T), np.arange(T)) + T - 1
    diag = np.bincount(d.ravel(), K.ravel()) / np.bincount(d.ravel())
    k0 = np.mean(np.diag(K))
    return float(np.max(np.abs(m - m.mean())) / np.sqrt(k0)), float(np.max(np.abs(K - diag[d])) / k0)


def time_mean(X):
    return X.mean(1)
