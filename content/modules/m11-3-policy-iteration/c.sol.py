import numpy as np


def _q(P, R, V, g):
    return R + g * np.einsum("ast,t->sa", P, V)


def vi(P, R, gamma, eps=1e-8):
    S = P.shape[1]
    V = np.zeros(S)
    it = 0
    while True:
        Vn = _q(P, R, V, gamma).max(1)
        it += 1
        done = np.abs(Vn - V).max() <= eps
        V = Vn
        if done:
            return V, _q(P, R, V, gamma).argmax(1), it, it * S


def pi_exact(P, R, gamma):
    S = P.shape[1]
    idx = np.arange(S)
    pi = np.zeros(S, int)
    it = 0
    while True:
        V = np.linalg.solve(np.eye(S) - gamma * P[pi, idx], R[idx, pi])
        it += 1
        new = _q(P, R, V, gamma).argmax(1)
        if np.array_equal(new, pi):
            return V, pi, it, it * (S * S + S)
        pi = new


def mpi(P, R, gamma, k, eps=1e-8):
    S = P.shape[1]
    idx = np.arange(S)
    V = np.zeros(S)
    it = backups = 0
    while True:
        Q = _q(P, R, V, gamma)
        pi, Vn = Q.argmax(1), Q.max(1)
        it += 1
        backups += S
        done = np.abs(Vn - V).max() <= eps
        V = Vn
        if done:
            return V, pi, it, backups
        Tp, Rp = P[pi, idx], R[idx, pi]
        for _ in range(k - 1):
            V = Rp + gamma * Tp @ V
            backups += S


rng = np.random.default_rng(0)
P = rng.random((3, 20, 20)); P /= P.sum(-1, keepdims=True)
print(pi_exact(P, rng.normal(size=(20, 3)), 0.95)[2])
