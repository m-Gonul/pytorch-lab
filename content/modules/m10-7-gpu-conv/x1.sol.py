import numpy as np


def winograd_f23(d, g):
    d0, d1, d2, d3 = d
    g0, g1, g2 = g
    m1 = (d0 - d2) * g0
    m2 = (d1 + d2) * (g0 + g1 + g2) / 2
    m3 = (d2 - d1) * (g0 - g1 + g2) / 2
    m4 = (d1 - d3) * g2
    return m1 + m2 + m3, m2 - m3 - m4


def conv1d_winograd(x, g):
    x = np.asarray(x, float)
    g0, g1, g2 = g
    L = len(x)
    i = np.arange(0, L - 2, 2)
    d0, d1, d2, d3 = x[i], x[i + 1], x[i + 2], x[i + 3]
    m1 = (d0 - d2) * g0
    m2 = (d1 + d2) * (g0 + g1 + g2) / 2
    m3 = (d2 - d1) * (g0 - g1 + g2) / 2
    m4 = (d1 - d3) * g2
    return np.stack([m1 + m2 + m3, m2 - m3 - m4], 1).ravel()


print(winograd_f23([1.0, 2, 3, 4], [1.0, 0, -1]))
