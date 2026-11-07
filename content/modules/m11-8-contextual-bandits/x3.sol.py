import math
import numpy as np


def make_logs(n, seed=0):
    rng = np.random.default_rng(seed)
    K, d = 5, 5
    theta = rng.uniform(0, 0.5, (K, d))
    X = rng.dirichlet(np.ones(d), n)
    A = rng.integers(0, K, n)
    R = (rng.random(n) < (theta[A] * X).sum(1)).astype(float)
    return X, A, R, theta


class _LinUCB:
    def __init__(self, K, d, alpha):
        self.Ainv = np.repeat(np.eye(d)[None], K, 0)
        self.b = np.zeros((K, d))
        self.alpha = alpha

    def choose(self, x):
        th = np.einsum("kij,kj->ki", self.Ainv, self.b)
        return int((th @ x + self.alpha * np.sqrt(np.einsum("i,kij,j->k", x, self.Ainv, x))).argmax())

    def update(self, a, x, r):
        Ax = self.Ainv[a] @ x
        self.Ainv[a] -= np.outer(Ax, Ax) / (1 + x @ Ax)
        self.b[a] += r * x


def replay_linucb(X, A, R, alpha=1.0):
    pol = _LinUCB(5, X.shape[1], alpha)
    tot, cnt = 0.0, 0
    for x, a, r in zip(X, A, R):
        if pol.choose(x) == a:
            pol.update(a, x, r)
            tot += r
            cnt += 1
    return tot / max(cnt, 1), cnt


def online_linucb(theta, T, seed=0, alpha=1.0):
    rng = np.random.default_rng(seed + 1000)
    pol = _LinUCB(5, theta.shape[1], alpha)
    tot = 0.0
    for _ in range(T):
        x = rng.dirichlet(np.ones(theta.shape[1]))
        a = pol.choose(x)
        r = float(rng.random() < theta[a] @ x)
        pol.update(a, x, r)
        tot += r
    return tot / T


X, A, R, th = make_logs(5000)
print(replay_linucb(X, A, R))
