import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def seasonal_difference(y, s):
    y = np.asarray(y, float)
    return y[s:] - y[:-s]


def seasonal_naive(y, s, h):
    y = np.asarray(y, float)
    k = np.arange(1, h + 1)
    return y[len(y) - s + (k - 1) % s]


def sarima_lite(y, s, p, h):
    y = np.asarray(y, float)
    z = seasonal_difference(np.diff(y), s)
    W = sliding_window_view(z, p + 1)
    phi = np.linalg.lstsq(W[:, :p][:, ::-1], W[:, p], rcond=None)[0]
    zz = list(z)
    yy = list(y)
    for _ in range(h):
        zn = float(np.dot(phi, zz[::-1][:p]))
        zz.append(zn)
        yy.append(zn + yy[-1] + yy[-s] - yy[-s - 1])
    return np.array(yy[len(y):])
