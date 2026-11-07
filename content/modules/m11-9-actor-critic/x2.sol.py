import numpy as np

W_STAR = np.array([[2.0, 0.0], [-1.0, 1.7], [-1.0, -1.7]])


def _softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def _test(W, seed):
    X = np.random.default_rng(seed + 999).normal(0, 1, (1000, 2))
    return float(((X @ W.T).argmax(1) == (X @ W_STAR.T).argmax(1)).mean())


def bc(n_samples, lr=0.5, seed=0):
    rng = np.random.default_rng(seed)
    W = np.zeros((3, 2))
    for _ in range(n_samples):
        x = rng.normal(0, 1, 2)
        a = int((W_STAR @ x).argmax())
        p = _softmax(W @ x)
        g = -p
        g[a] += 1
        W += lr * np.outer(g, x)
    return _test(W, seed)


def pg(n_samples, lr=0.5, seed=0):
    rng = np.random.default_rng(seed)
    W = np.zeros((3, 2))
    b = 0.0
    for t in range(n_samples):
        x = rng.normal(0, 1, 2)
        p = _softmax(W @ x)
        a = rng.choice(3, p=p)
        r = float(a == int((W_STAR @ x).argmax()))
        g = -p
        g[a] += 1
        W += lr * (r - b) * np.outer(g, x)
        b += (r - b) / (t + 1)
    return _test(W, seed)


print(bc(500), pg(500))
