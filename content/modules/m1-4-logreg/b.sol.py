import math


def sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)               # z < 0 → e ∈ (0, 1), taşmaz
    return e / (1.0 + e)


def bce(p, y, eps=1e-12):
    p = min(max(p, eps), 1 - eps)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def grad(X, y, w, b):
    """→ (gw: liste, gb, ortalama kayıp)"""
    n, d = len(X), len(w)
    gw, gb, L = [0.0] * d, 0.0, 0.0
    for xi, yi in zip(X, y):
        z = sum(wj * xj for wj, xj in zip(w, xi)) + b
        p = sigmoid(z)
        L += bce(p, yi) / n
        for j in range(d):
            gw[j] += (p - yi) * xi[j] / n
        gb += (p - yi) / n
    return gw, gb, L


def fit(X, y, lr=0.5, steps=1000):
    w, b = [0.0] * len(X[0]), 0.0
    for _ in range(steps):
        gw, gb, _ = grad(X, y, w, b)
        w = [wj - lr * g for wj, g in zip(w, gw)]
        b -= lr * gb
    return w, b


X = [[-1.0], [0.0], [2.0]]
y = [0, 1, 1]
print(grad(X, y, [1.0], 0.0))      # ([-0.169…], -0.1167…, 0.3777…)
print(sigmoid(-1000), sigmoid(1000))
