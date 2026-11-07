import numpy as np


def vi(P, R, gamma, eps=1e-8):
    ...


def pi_exact(P, R, gamma):
    ...


def mpi(P, R, gamma, k, eps=1e-8):
    ...


rng = np.random.default_rng(0)
P = rng.random((3, 20, 20)); P /= P.sum(-1, keepdims=True)
print(pi_exact(P, rng.normal(size=(20, 3)), 0.95)[2])
