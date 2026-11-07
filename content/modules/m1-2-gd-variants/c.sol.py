import numpy as np


def train(X, y, lr, batch_size, epochs, beta=0.0, seed=0):
    """→ (w, b, losses: (epochs,)). En fazla 2 katman döngü (epoch, batch)."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    n, d = X.shape
    rng = np.random.default_rng(seed)
    w, b = np.zeros(d), 0.0
    vw, vb = np.zeros(d), 0.0
    losses = []
    for _ in range(epochs):
        perm = rng.permutation(n)
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            e = X[idx] @ w + b - y[idx]
            gw = 2 * X[idx].T @ e / len(idx)
            gb = 2 * e.mean()
            vw = beta * vw + gw
            vb = beta * vb + gb
            w = w - lr * vw
            b = b - lr * vb
        r = X @ w + b - y
        losses.append(float(r @ r / n))
    return w, b, np.array(losses)


rng = np.random.default_rng(0)
X = rng.normal(size=(500, 3))
y = X @ np.array([1.0, -2.0, 0.5]) + 1.0 + 0.1 * rng.normal(size=500)
w, b, losses = train(X, y, lr=0.05, batch_size=32, epochs=30, beta=0.9)
print(w, b, losses[-1])
