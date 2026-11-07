import numpy as np


def _mdp(rng, S, A):
    P = rng.random((A, S, S)) ** 4; P /= P.sum(-1, keepdims=True)
    return P, rng.normal(size=(S, A))


def pi_iterations(S, A, gamma, trials, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    idx = np.arange(S)
    for _ in range(trials):
        P, R = _mdp(rng, S, A)
        pi = np.zeros(S, int); n = 0
        while True:
            V = np.linalg.solve(np.eye(S) - gamma * P[pi, idx], R[idx, pi]); n += 1
            new = (R + gamma * np.einsum("ast,t->sa", P, V)).argmax(1)
            if np.array_equal(new, pi):
                break
            pi = new
        out.append(n)
    return out


def vi_iterations(S, A, gamma, trials, seed=0, eps=1e-8):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(trials):
        P, R = _mdp(rng, S, A)
        V = np.zeros(S); n = 0
        while True:
            Vn = (R + gamma * np.einsum("ast,t->sa", P, V)).max(1); n += 1
            if np.abs(Vn - V).max() <= eps:
                break
            V = Vn
        out.append(n)
    return out


print(pi_iterations(10, 2, 0.9, 3))
