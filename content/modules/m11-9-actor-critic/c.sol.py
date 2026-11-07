import numpy as np


def softmax_policy(theta):
    e = np.exp(theta - theta.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def evaluate(P, R, pi, gamma):
    S = R.shape[0]
    Ppi = np.einsum("sa,ast->st", pi, P)
    rpi = (pi * R).sum(1)
    V = np.linalg.solve(np.eye(S) - gamma * Ppi, rpi)
    return V, R + gamma * np.einsum("ast,t->sa", P, V)


def J(theta, P, R, d0, gamma):
    return float(d0 @ evaluate(P, R, softmax_policy(theta), gamma)[0])


def pg_theorem(theta, P, R, d0, gamma):
    pi = softmax_policy(theta)
    V, Q = evaluate(P, R, pi, gamma)
    Ppi = np.einsum("sa,ast->st", pi, P)
    d = np.linalg.solve(np.eye(len(d0)) - gamma * Ppi.T, d0)
    return d[:, None] * pi * (Q - V[:, None])


def finite_diff(theta, P, R, d0, gamma, eps=1e-6):
    g = np.zeros_like(theta)
    for idx in np.ndindex(theta.shape):
        e = np.zeros_like(theta)
        e[idx] = eps
        g[idx] = (J(theta + e, P, R, d0, gamma) - J(theta - e, P, R, d0, gamma)) / (2 * eps)
    return g


rng = np.random.default_rng(0)
P = rng.random((2, 4, 4)); P /= P.sum(-1, keepdims=True)
R = rng.random((4, 2)); d0 = np.full(4, 0.25); th = rng.normal(0, 1, (4, 2))
print(np.abs(pg_theorem(th, P, R, d0, 0.9) - finite_diff(th, P, R, d0, 0.9)).max())
