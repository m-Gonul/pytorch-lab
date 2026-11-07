import numpy as np


def per_sample_grads(w, X, y):
    ...


def hessian(w, X):
    ...


def hvp(w, X, v):
    ...


rng = np.random.default_rng(0)
X, y, w = rng.normal(size=(6, 3)), rng.integers(0, 2, 6), rng.normal(size=3)
print(per_sample_grads(w, X, y).mean(0))
print(hvp(w, X, np.ones(3)), hessian(w, X) @ np.ones(3))
