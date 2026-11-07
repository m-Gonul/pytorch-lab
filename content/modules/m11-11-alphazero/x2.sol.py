import numpy as np

LINES = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]


def _term(b):
    for l in LINES:
        s = b[l[0]] + b[l[1]] + b[l[2]]
        if s == 3 or s == -3:
            return s // 3
    return 0 if all(b) else None


def _dataset():
    memo = {}

    def val(b, p):
        k = (b, p)
        if k in memo:
            return memo[k]
        t = _term(b)
        if t is not None:
            return t
        best = None
        for m in range(9):
            if b[m] == 0:
                v = val(b[:m] + (p,) + b[m + 1:], -p)
                best = v if best is None or v * p > best * p else best
        memo[k] = best
        return best

    val(tuple([0] * 9), 1)
    X = np.array([np.r_[np.array(b) * p == 1, np.array(b) * p == -1] for (b, p) in memo], float)
    y = np.array([v * p for (b, p), v in memo.items()], float)
    return X, y


def train_value(X, y, iters=3000, lr=0.05, seed=0):
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, 0.3, (18, 32)); b1 = np.zeros(32)
    w2 = rng.normal(0, 0.3, 32); b2 = 0.0
    n = len(y)
    for _ in range(iters):
        h = np.tanh(X @ W1 + b1)
        out = np.tanh(h @ w2 + b2)
        g = 2 * (out - y) / n * (1 - out ** 2)
        gh = np.outer(g, w2) * (1 - h ** 2)
        w2 -= lr * h.T @ g; b2 -= lr * g.sum()
        W1 -= lr * X.T @ gh; b1 -= lr * gh.sum(0)
    return W1, b1, w2, b2


def predict(params, X):
    W1, b1, w2, b2 = params
    return np.tanh(np.tanh(X @ W1 + b1) @ w2 + b2)


def _augment(X, y):
    g = np.arange(9).reshape(3, 3)
    P = np.array([f(np.rot90(g, k)).ravel() for k in range(4) for f in (lambda x: x, np.fliplr)])
    Xa = np.concatenate([X[:, :9][:, P], X[:, 9:][:, P]], 2).transpose(1, 0, 2).reshape(-1, 18)
    return Xa, np.tile(y, 8)


def experiment(n_train=150, seed=0):
    X, y = _dataset()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), n_train, replace=False)
    test = np.setdiff1d(np.arange(len(y)), idx)
    p1 = train_value(X[idx], y[idx], seed=seed)
    Xa, ya = _augment(X[idx], y[idx])
    p2 = train_value(Xa, ya, seed=seed)
    mse = lambda p: float(np.mean((predict(p, X[test]) - y[test]) ** 2))
    return mse(p1), mse(p2)


print(experiment())
