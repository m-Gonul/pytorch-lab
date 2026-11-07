import numpy as np
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier


def shift_augment(X):
    I = X.reshape(-1, 8, 8)
    outs = [I]
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        S = np.zeros_like(I)
        ys = slice(max(dy, 0), 8 + min(dy, 0))
        yd = slice(max(-dy, 0), 8 + min(-dy, 0))
        xs = slice(max(dx, 0), 8 + min(dx, 0))
        xd = slice(max(-dx, 0), 8 + min(-dx, 0))
        S[:, ys, xs] = I[:, yd, xd]
        outs.append(S)
    return np.concatenate(outs).reshape(-1, 64)


def few_shot_accuracy(augment, seed, n_per_class=3):
    X, y = load_digits(return_X_y=True)
    rng = np.random.default_rng(seed)
    tr = np.concatenate([rng.choice(np.flatnonzero(y == c), n_per_class, replace=False) for c in range(10)])
    te = np.setdiff1d(np.arange(len(y)), tr)
    Xtr, ytr = X[tr], y[tr]
    if augment:
        Xtr, ytr = shift_augment(Xtr), np.tile(ytr, 5)
    return float(KNeighborsClassifier(1).fit(Xtr, ytr).score(X[te], y[te]))


print(few_shot_accuracy(False, 0), few_shot_accuracy(True, 0))
