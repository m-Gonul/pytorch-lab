import numpy as np


def sigmoid_np(z):
    z = np.asarray(z, dtype=float)
    q = np.exp(-np.abs(z))                     # her zaman ≤ 1, taşmaz
    return np.where(z >= 0, 1 / (1 + q), q / (1 + q))


def fit_logreg(X, y, lr=0.5, steps=500):
    """→ (w, b, losses). Yalnız adım döngüsü."""
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    losses = []
    for _ in range(steps):
        z = X @ w + b
        p = sigmoid_np(z)
        losses.append(float(np.mean(np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z))))))
        r = p - y
        w = w - lr * X.T @ r / n
        b = b - lr * r.mean()
    return w, b, np.array(losses)


def predict(X, w, b):
    return (np.asarray(X) @ w + b >= 0).astype(int)


rng = np.random.default_rng(0)
X = np.concatenate([rng.normal(-1.5, 1, (100, 2)), rng.normal(1.5, 1, (100, 2))])
y = np.repeat([0, 1], 100)
w, b, losses = fit_logreg(X, y)
print("doğruluk:", (predict(X, w, b) == y).mean(), "son kayıp:", losses[-1])
