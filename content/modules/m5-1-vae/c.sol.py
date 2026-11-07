import numpy as np


def ae_grads(X, We, Wd):
    n = X.shape[0]
    Z = X @ We
    R = Z @ Wd - X
    L = (R ** 2).sum() / n
    G = 2 * R / n
    return L, X.T @ (G @ Wd.T), Z.T @ G


def train_linear_ae(X, k, steps=3000, lr=0.02, seed=0):
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    We = 0.1 * rng.normal(size=(d, k))
    Wd = 0.1 * rng.normal(size=(k, d))
    for _ in range(steps):
        L, dWe, dWd = ae_grads(X, We, Wd)
        We -= lr * dWe
        Wd -= lr * dWd
    return We, Wd, ae_grads(X, We, Wd)[0]


def pca_error(X, k):
    ev = np.linalg.eigvalsh(X.T @ X / X.shape[0])
    return ev[: X.shape[1] - k].sum()


rng = np.random.default_rng(0)
X = rng.normal(size=(200, 5)) * np.array([3, 2, 1, 0.3, 0.1])
X = X @ np.linalg.qr(rng.normal(size=(5, 5)))[0]
X -= X.mean(0)
We, Wd, L = train_linear_ae(X, 2)
print(L, pca_error(X, 2))
