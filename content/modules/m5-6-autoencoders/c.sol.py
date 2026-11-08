import numpy as np


def tied_sparse_ae(X, W, b, c, rho, beta):
    N = len(X)
    z = 1 / (1 + np.exp(-(X @ W + b)))
    Xh = z @ W.T + c
    rh = z.mean(0)
    L = 0.5 * np.sum((Xh - X) ** 2) / N + beta * np.sum(rho * np.log(rho / rh) + (1 - rho) * np.log((1 - rho) / (1 - rh)))
    dXh = (Xh - X) / N
    dz = dXh @ W + beta * (-rho / rh + (1 - rho) / (1 - rh)) / N
    da = dz * z * (1 - z)
    dW = X.T @ da + dXh.T @ z
    return L, dW, da.sum(0), dXh.sum(0)


def train(X, k, rho, beta, lr=0.5, iters=1500, seed=0):
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    W = rng.normal(0, 0.1, (d, k))
    b, c = np.zeros(k), X.mean(0)
    for _ in range(iters):
        _, dW, db, dc = tied_sparse_ae(X, W, b, c, rho, beta)
        W -= lr * dW
        b -= lr * db
        c -= lr * dc
    return W, b, c
