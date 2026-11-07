import numpy as np


def softmax_rows(Z):
    ...


def softmax_loss(W, X, y, reg=0.0):
    """→ (loss, dW). Döngü yok."""
    ...


def fit_softmax(X, y, K, lr=0.5, steps=300):
    ...


rng = np.random.default_rng(0)
centers = np.array([[0, 3], [-3, -2], [3, -2]])
y = np.repeat(np.arange(3), 50)
X = np.column_stack([centers[y] + rng.normal(size=(150, 2)), np.ones(150)])
W = fit_softmax(X, y, 3)
print("doğruluk:", (np.argmax(X @ W, 1) == y).mean())
