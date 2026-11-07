import numpy as np


def _V(P, R, pi, gamma):
    S = R.shape[0]
    Ppi = np.einsum("sa,ast->st", pi, P)
    V = np.linalg.solve(np.eye(S) - gamma * Ppi, (pi * R).sum(1))
    Q = R + gamma * np.einsum("ast,t->sa", P, V)
    return V, Q, Ppi


def eta(P, R, d0, pi, gamma):
    return float(d0 @ _V(P, R, pi, gamma)[0])


def surrogate_L(P, R, d0, pi, pi_new, gamma):
    V, Q, Ppi = _V(P, R, pi, gamma)
    rho = np.linalg.solve(np.eye(len(d0)) - gamma * Ppi.T, d0)
    return float(d0 @ V + rho @ (pi_new * (Q - V[:, None])).sum(1))


def kl_max(pi, pi_new):
    return float((pi * (np.log(pi) - np.log(pi_new))).sum(1).max())


def bound(P, R, d0, pi, pi_new, gamma):
    V, Q, _ = _V(P, R, pi, gamma)
    eps = np.abs(Q - V[:, None]).max()
    return surrogate_L(P, R, d0, pi, pi_new, gamma) - 4 * eps * gamma / (1 - gamma) ** 2 * kl_max(pi, pi_new)


rng = np.random.default_rng(0)
P = rng.random((3, 6, 6)); P /= P.sum(-1, keepdims=True); R = rng.random((6, 3)); d0 = np.full(6, 1 / 6)
pi = np.full((6, 3), 1 / 3)
print(eta(P, R, d0, pi, 0.9))
