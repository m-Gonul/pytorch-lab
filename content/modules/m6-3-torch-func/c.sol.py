import numpy as np


def _p(w, X):
    return 1 / (1 + np.exp(-(X @ w)))


def per_sample_grads(w, X, y):
    return (_p(w, X) - y)[:, None] * X


def hessian(w, X):
    p = _p(w, X)
    return (X * (p * (1 - p))[:, None]).T @ X / X.shape[0]


def hvp(w, X, v):
    p = _p(w, X)
    return X.T @ (p * (1 - p) * (X @ v)) / X.shape[0]


rng = np.random.default_rng(0)
X, y, w = rng.normal(size=(6, 3)), rng.integers(0, 2, 6), rng.normal(size=3)
print(per_sample_grads(w, X, y).mean(0))
print(hvp(w, X, np.ones(3)), hessian(w, X) @ np.ones(3))
