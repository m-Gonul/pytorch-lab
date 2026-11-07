import numpy as np


def relu_features(X, W):
    return np.maximum(0.0, X @ W)


def min_norm_fit(F, y):
    return np.linalg.pinv(F) @ y


def double_descent(n=100, d=10, ps=(10, 25, 50, 75, 90, 100, 110, 125, 150, 200, 400, 1000, 3000), trials=20, sigma=0.3, seed=0):
    rng = np.random.default_rng(seed)
    beta = rng.normal(size=d) / np.sqrt(d)
    res = {p: [] for p in ps}
    for _ in range(trials):
        X = rng.normal(size=(n, d))
        y = np.tanh(2 * X @ beta) + sigma * rng.normal(size=n)
        Xt = rng.normal(size=(2000, d))
        yt = np.tanh(2 * Xt @ beta)
        for p in ps:
            W = rng.normal(size=(d, p)) / np.sqrt(d)
            coef = min_norm_fit(relu_features(X, W), y)
            res[p].append(np.mean((relu_features(Xt, W) @ coef - yt) ** 2))
    return {p: float(np.median(v)) for p, v in res.items()}


def relu_interpolant(knots, values):
    k, v = np.asarray(knots, float), np.asarray(values, float)
    slopes = np.diff(v) / np.diff(k)
    c1 = slopes[0]
    c0 = v[0] - c1 * k[0]
    a = np.diff(slopes)
    return a, c0, c1


print(relu_interpolant(np.array([0.0, 1, 2, 3]), np.array([0.0, 1, 0, 2])))
