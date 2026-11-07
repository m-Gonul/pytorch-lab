import numpy as np


def q_values(P, R, V, gamma):
    return R + gamma * np.einsum("ast,t->sa", P, V)


def value_iteration(P, R, gamma, eps=1e-8, V0=None):
    V = np.zeros(P.shape[1]) if V0 is None else np.array(V0, float)
    n = 0
    while True:
        Vn = q_values(P, R, V, gamma).max(1)
        n += 1
        d = np.abs(Vn - V).max()
        V = Vn
        if d <= eps:
            break
    return V, q_values(P, R, V, gamma).argmax(1), n, d


def policy_eval(P, R, pi, gamma):
    S = P.shape[1]
    idx = np.arange(S)
    return np.linalg.solve(np.eye(S) - gamma * P[pi, idx], R[idx, pi])


def bound(gamma, diff):
    return gamma / (1 - gamma) * diff


rng = np.random.default_rng(0)
P = rng.random((2, 4, 4)); P /= P.sum(-1, keepdims=True)
print(value_iteration(P, rng.random((4, 2)), 0.9)[2])
