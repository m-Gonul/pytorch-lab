import numpy as np


def _finite_first(P, R, VT, gamma, h):
    V = np.array(VT, float)
    pi = None
    for _ in range(h):
        Q = R + gamma * np.einsum("ast,t->sa", P, V)
        pi, V = Q.argmax(1), Q.max(1)
    return pi


def stationary_policy_horizon(P, R, VT, gamma, H):
    pols = [_finite_first(P, R, VT, gamma, h) for h in range(1, H + 1)]
    last = pols[-1]
    k = H
    while k > 1 and np.array_equal(pols[k - 2], last):
        k -= 1
    return pols, k


def policy_vs_gamma(P, R, gammas):
    out = []
    for g in gammas:
        V = np.zeros(P.shape[1])
        for _ in range(3000):
            V = (R + g * np.einsum("ast,t->sa", P, V)).max(1)
        out.append((R + g * np.einsum("ast,t->sa", P, V)).argmax(1))
    return out


print("ufuk")
