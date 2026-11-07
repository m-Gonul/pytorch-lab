import numpy as np


def q_values(P, R, V, gamma):
    ...


def value_iteration(P, R, gamma, eps=1e-8, V0=None):
    ...


def policy_eval(P, R, pi, gamma):
    ...


def bound(gamma, diff):
    ...


rng = np.random.default_rng(0)
P = rng.random((2, 4, 4)); P /= P.sum(-1, keepdims=True)
print(value_iteration(P, rng.random((4, 2)), 0.9)[2])
