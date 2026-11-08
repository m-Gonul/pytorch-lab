import numpy as np


def rrelu(z, train, rng, l=1 / 8, u=1 / 3):
    a = rng.uniform(l, u, z.shape) if train else (l + u) / 2
    return np.where(z >= 0, z, a * z)


def prelu_grads(z, alpha, gout):
    return gout * np.where(z > 0, 1.0, alpha), float(np.sum(gout * np.where(z > 0, 0.0, z)))


def elu(z, a=1.0):
    return np.where(z > 0, z, a * (np.exp(np.minimum(z, 0)) - 1))


def mish(z):
    return z * np.tanh(np.logaddexp(0, z))


def dead_fraction(X, W, b):
    return float(((X @ W + b) <= 0).all(0).mean())
