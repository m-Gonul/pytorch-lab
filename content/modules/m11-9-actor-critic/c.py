import numpy as np


def softmax_policy(theta):
    ...


def evaluate(P, R, pi, gamma):
    ...


def J(theta, P, R, d0, gamma):
    ...


def pg_theorem(theta, P, R, d0, gamma):
    ...


def finite_diff(theta, P, R, d0, gamma, eps=1e-6):
    ...


rng = np.random.default_rng(0)
P = rng.random((2, 4, 4)); P /= P.sum(-1, keepdims=True)
R = rng.random((4, 2)); d0 = np.full(4, 0.25); th = rng.normal(0, 1, (4, 2))
print(np.abs(pg_theorem(th, P, R, d0, 0.9) - finite_diff(th, P, R, d0, 0.9)).max())
