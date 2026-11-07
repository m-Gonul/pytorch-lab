import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def acf(y, maxlag):
    z = np.asarray(y, float) - np.mean(y)
    n = len(z)
    g = np.correlate(z, z, "full")[n - 1:n + maxlag] / n
    return g / g[0]


def lag_matrix(y, p):
    W = sliding_window_view(np.asarray(y, float), p + 1)
    return W[:, :p][:, ::-1], W[:, p]


def fit_ar_ls(y, p):
    X, t = lag_matrix(y, p)
    return np.linalg.lstsq(X, t, rcond=None)[0]


def fit_ar_yw(y, p):
    r = acf(y, p)
    idx = np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
    return np.linalg.solve(r[idx], r[1:p + 1])


def pacf(y, maxlag):
    out = [1.0]
    for k in range(1, maxlag + 1):
        out.append(fit_ar_yw(y, k)[-1])
    return np.array(out)


def forecast_ar(y, phi, h):
    hist = list(np.asarray(y, float)[-len(phi):])
    out = []
    for _ in range(h):
        v = float(np.dot(phi, hist[::-1][:len(phi)]))
        out.append(v)
        hist.append(v)
    return np.array(out)
