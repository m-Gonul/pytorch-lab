import numpy as np


def _s(z):
    return 1 / (1 + np.exp(-z))


def peephole_step(x, h, c, W, U, b, P):
    a = x @ W.T + h @ U.T + b
    ai, af, ag, ao = np.split(a, 4, axis=-1)
    i = _s(ai + P[0] * c)
    f = _s(af + P[1] * c)
    c2 = f * c + i * np.tanh(ag)
    o = _s(ao + P[2] * c2)
    return o * np.tanh(c2), c2


def cifg_step(x, h, c, W, U, b):
    a = x @ W.T + h @ U.T + b
    af, ag, ao = np.split(a, 3, axis=-1)
    f = _s(af)
    c2 = f * c + (1 - f) * np.tanh(ag)
    return _s(ao) * np.tanh(c2), c2
