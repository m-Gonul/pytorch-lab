import numpy as np


def fit_model(S_, A_, R_, S2_, nS, nA):
    cnt = np.zeros((nA, nS, nS))
    np.add.at(cnt, (A_, S_, S2_), 1)
    n = cnt.sum(-1)
    rs = np.zeros((nS, nA))
    np.add.at(rs, (S_, A_), R_)
    seen = n > 0
    P = np.where(seen[..., None], cnt / np.maximum(n, 1)[..., None], np.eye(nS)[None])
    return P, np.where(seen.T, rs / np.maximum(n.T, 1), 0.0)


def plan(P, R, gamma, n_iter=500):
    Q = np.zeros_like(R)
    for _ in range(n_iter):
        Q = R + gamma * np.einsum("ast,t->sa", P, Q.max(1))
    return Q


def q_learning_replay(S_, A_, R_, S2_, nS, nA, gamma, epochs=1, alpha=0.1):
    Q = np.zeros((nS, nA))
    for _ in range(epochs):
        for s, a, r, s2 in zip(S_, A_, R_, S2_):
            Q[s, a] += alpha * (r + gamma * Q[s2].max() - Q[s, a])
    return Q


def policy_value(P, R, pi, gamma):
    nS = R.shape[0]
    Ppi = P[pi, np.arange(nS)]
    return np.linalg.solve(np.eye(nS) - gamma * Ppi, R[np.arange(nS), pi])


rng = np.random.default_rng(0)
P = rng.random((2, 5, 5)); P /= P.sum(-1, keepdims=True); R = rng.random((5, 2))
S_ = rng.integers(0, 5, 100); A_ = rng.integers(0, 2, 100)
S2_ = np.array([rng.choice(5, p=P[a, s]) for s, a in zip(S_, A_)]); R_ = R[S_, A_]
Ph, Rh = fit_model(S_, A_, R_, S2_, 5, 2)
print(np.abs(Ph - P).max())
