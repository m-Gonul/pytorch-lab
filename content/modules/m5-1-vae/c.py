import numpy as np


def ae_grads(X, We, Wd):
    ...


def train_linear_ae(X, k, steps=3000, lr=0.02, seed=0):
    ...


def pca_error(X, k):
    ...


rng = np.random.default_rng(0)
X = rng.normal(size=(200, 5)) * np.array([3, 2, 1, 0.3, 0.1])
X = X @ np.linalg.qr(rng.normal(size=(5, 5)))[0]
X -= X.mean(0)
We, Wd, L = train_linear_ae(X, 2)
print(L, pca_error(X, 2))
