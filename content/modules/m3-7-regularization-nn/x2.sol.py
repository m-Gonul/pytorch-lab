import numpy as np


def max_norm(W, r, axis=0):
    n = np.linalg.norm(W, axis=axis, keepdims=True)
    return W * np.minimum(1.0, r / np.maximum(n, 1e-12))


def train(X, y, hidden, lr, epochs, r, seed):
    rng = np.random.default_rng(seed)
    d, C = X.shape[1], int(y.max()) + 1
    W1, b1 = rng.normal(0, np.sqrt(2 / d), (d, hidden)), np.zeros(hidden)
    W2, b2 = rng.normal(0, np.sqrt(1 / hidden), (hidden, C)), np.zeros(C)
    for _ in range(epochs):
        p = rng.permutation(len(X))
        for i in range(0, len(X), 32):
            xb, yb = X[p[i:i + 32]], y[p[i:i + 32]]
            h = np.maximum(0, xb @ W1 + b1)
            z = h @ W2 + b2
            P = np.exp(z - z.max(1, keepdims=True))
            P /= P.sum(1, keepdims=True)
            P[np.arange(len(yb)), yb] -= 1
            dz = P / len(yb)
            dh = (dz @ W2.T) * (h > 0)
            W2 -= lr * h.T @ dz
            b2 -= lr * dz.sum(0)
            W1 -= lr * xb.T @ dh
            b1 -= lr * dh.sum(0)
            if r is not None:
                W1, W2 = max_norm(W1, r), max_norm(W2, r)
    return W1, b1, W2, b2
