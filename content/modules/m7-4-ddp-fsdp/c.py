import numpy as np


def worker_grads(w, X, y, K):
    ...


def allreduce_mean(grads):
    ...


def adam_update(w, g, m, v, t, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8):
    ...


def zero1_step(w, g, states, t, lr=1e-2):
    ...


rng = np.random.default_rng(0)
X, y, w = rng.normal(size=(64, 6)), rng.normal(size=64), rng.normal(size=6)
g = allreduce_mean(worker_grads(w, X, y, 4))
print(g - 2 * X.T @ (X @ w - y) / 64)
