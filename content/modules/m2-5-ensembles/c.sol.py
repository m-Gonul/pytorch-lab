import numpy as np


def best_reg_stump(X, r):
    """→ (j, t, sol, sağ) ya da None. Döngü yok."""
    n, d = X.shape
    order = np.argsort(X, axis=0, kind="stable")
    Xs = np.take_along_axis(X, order, 0)
    Rs = np.asarray(r, float)[order]                         # (n, d)
    SL = np.cumsum(Rs, 0)[:-1]                               # (n-1, d)
    SR = Rs.sum(0) - SL
    nl = np.arange(1, n)[:, None]
    gain = SL ** 2 / nl + SR ** 2 / (n - nl)
    gain = np.where(Xs[:-1] == Xs[1:], -np.inf, gain)
    if not np.isfinite(gain).any():
        return None
    g = gain.T
    j, i = np.unravel_index(np.argmax(g), g.shape)
    return int(j), float((Xs[i, j] + Xs[i + 1, j]) / 2), float(SL[i, j] / (i + 1)), float(SR[i, j] / (n - i - 1))


def stump_np(st, X):
    j, t, l, r = st
    return np.where(X[:, j] <= t, l, r)


def gbm_np(X, y, rounds, lr):
    F0 = float(np.mean(y))
    F = np.full(len(y), F0)
    stumps = []
    for _ in range(rounds):
        st = best_reg_stump(X, y - F)
        if st is None:
            break
        stumps.append(st)
        F = F + lr * stump_np(st, X)
    return F0, lr, stumps


def gbm_predict_np(model, X):
    F0, lr, stumps = model
    F = np.full(X.shape[0], F0)
    for st in stumps:
        F = F + lr * stump_np(st, X)
    return F


rng = np.random.default_rng(0)
X = rng.uniform(-2, 2, size=(400, 2))
y = np.sin(2 * X[:, 0]) + X[:, 1] ** 2
m = gbm_np(X, y, 100, 0.2)
print(np.mean((gbm_predict_np(m, X) - y) ** 2))
