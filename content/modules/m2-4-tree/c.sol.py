import numpy as np


def best_split_np(X, y, C):
    """→ (j, t, skor) ya da None. Döngü yok."""
    X = np.asarray(X, float)
    n, d = X.shape
    order = np.argsort(X, axis=0, kind="stable")
    Xs = np.take_along_axis(X, order, 0)                     # (n, d)
    Y = np.eye(C)[np.asarray(y)[order]]                      # (n, d, C)
    left = np.cumsum(Y, axis=0)[:-1]                         # (n-1, d, C)
    right = Y.sum(0) - left
    nl = np.arange(1, n)[:, None]                            # (n-1, 1)
    nr = n - nl
    gl = 1 - ((left / nl[..., None]) ** 2).sum(2)
    gr = 1 - ((right / nr[..., None]) ** 2).sum(2)
    score = nl / n * gl + nr / n * gr                        # (n-1, d)
    score = np.where(Xs[:-1] == Xs[1:], np.inf, score)
    if not np.isfinite(score).any():
        return None
    s = score.T                                              # (d, n-1): özellik önce, eşik artan
    j, i = np.unravel_index(np.argmin(s), s.shape)
    return int(j), float((Xs[i, j] + Xs[i + 1, j]) / 2), float(s[j, i])


print(best_split_np(np.array([[1.0], [2], [3], [4], [5], [6]]), np.array([0, 0, 1, 1, 1, 0]), 2))
