import numpy as np


def eta(P, R, d0, pi, gamma):
    ...


def surrogate_L(P, R, d0, pi, pi_new, gamma):
    ...


def kl_max(pi, pi_new):
    ...


def bound(P, R, d0, pi, pi_new, gamma):
    ...


rng = np.random.default_rng(0)
P = rng.random((3, 6, 6)); P /= P.sum(-1, keepdims=True); R = rng.random((6, 3)); d0 = np.full(6, 1 / 6)
pi = np.full((6, 3), 1 / 3)
print(eta(P, R, d0, pi, 0.9))
