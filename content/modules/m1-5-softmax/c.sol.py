import numpy as np


def softmax_rows(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def softmax_loss(W, X, y, reg=0.0):
    """→ (loss, dW). Döngü yok."""
    n = X.shape[0]
    Z = X @ W
    Zs = Z - Z.max(axis=1, keepdims=True)
    logP = Zs - np.log(np.exp(Zs).sum(axis=1, keepdims=True))
    loss = -logP[np.arange(n), y].mean() + reg * np.sum(W * W)
    G = np.exp(logP)
    G[np.arange(n), y] -= 1
    dW = X.T @ G / n + 2 * reg * W
    return loss, dW


def fit_softmax(X, y, K, lr=0.5, steps=300):
    W = np.zeros((X.shape[1], K))
    for _ in range(steps):
        _, dW = softmax_loss(W, X, y)
        W -= lr * dW
    return W


rng = np.random.default_rng(0)
centers = np.array([[0, 3], [-3, -2], [3, -2]])
y = np.repeat(np.arange(3), 50)
X = np.column_stack([centers[y] + rng.normal(size=(150, 2)), np.ones(150)])
W = fit_softmax(X, y, 3)
print("doğruluk:", (np.argmax(X @ W, 1) == y).mean())
