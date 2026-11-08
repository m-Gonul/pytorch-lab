import numpy as np


def bon_dist(p, r, n):
    o = np.argsort(r)
    F = np.cumsum(p[o])
    Fm = np.concatenate([[0.0], F[:-1]])
    q = np.empty_like(p)
    q[o] = F ** n - Fm ** n
    return q


def kl(q, p):
    m = q > 0
    return float(np.sum(q[m] * np.log(q[m] / p[m])))
