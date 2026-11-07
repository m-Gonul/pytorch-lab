import numpy as np


def _V(P, R, g, pi):
    S = len(pi); idx = np.arange(S)
    return np.linalg.solve(np.eye(S) - g * P[pi, idx], R[idx, pi])


def q_from_v(P, R, gamma, V):
    return R + gamma * np.einsum("ast,t->sa", P, V)


def check_improvement(P, R, gamma, pi, seed=0):
    V = _V(P, R, gamma, pi)
    Q = q_from_v(P, R, gamma, V)
    new = Q.argmax(1)
    return _V(P, R, gamma, new) - V, Q[np.arange(len(pi)), new] - V


def random_improving_policy(P, R, gamma, pi, seed=0):
    rng = np.random.default_rng(seed)
    V = _V(P, R, gamma, pi)
    Q = q_from_v(P, R, gamma, V)
    cand = np.argwhere(Q > V[:, None] + 1e-12)
    new = np.array(pi).copy()
    if len(cand):
        s, a = cand[rng.integers(len(cand))]
        new[s] = a
    return new


print("iyileştirme")
