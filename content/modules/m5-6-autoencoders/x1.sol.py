import numpy as np


def dae_optimal(Sigma, s2):
    return Sigma @ np.linalg.inv(Sigma + s2 * np.eye(len(Sigma)))


def train_linear_dae(X, s2, lr, iters, seed):
    rng = np.random.default_rng(seed)
    N, d = X.shape
    W = np.zeros((d, d))
    for _ in range(iters):
        Xn = X + rng.normal(0, np.sqrt(s2), X.shape)
        W -= lr * (Xn @ W.T - X).T @ Xn / N
    return W
