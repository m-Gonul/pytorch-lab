import numpy as np


def svm_loss(W, X, y, reg=0.0, delta=1.0):
    """→ (loss, dW). Döngü yok."""
    n = X.shape[0]
    S = X @ W
    sy = S[np.arange(n), y][:, None]
    M = np.maximum(0, S - sy + delta)
    M[np.arange(n), y] = 0
    loss = M.sum() / n + reg * np.sum(W * W)
    B = (M > 0).astype(float)
    B[np.arange(n), y] = -B.sum(1)
    return loss, X.T @ B / n + 2 * reg * W


rng = np.random.default_rng(0)
W = rng.normal(size=(5, 3)) * 0.01
X = rng.normal(size=(10, 5))
y = rng.integers(0, 3, 10)
print(svm_loss(W, X, y, reg=0.1))
