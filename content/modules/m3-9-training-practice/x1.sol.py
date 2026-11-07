import numpy as np


def topk_features(X, y, k):
    c = np.abs(((X - X.mean(0)) * (y - y.mean())[:, None]).sum(0))
    return np.argsort(c)[-k:]


def nearest_centroid_acc(Xtr, ytr, Xte, yte):
    m0, m1 = Xtr[ytr == 0].mean(0), Xtr[ytr == 1].mean(0)
    pred = (((Xte - m1) ** 2).sum(1) < ((Xte - m0) ** 2).sum(1)).astype(int)
    return float((pred == yte).mean())


def cv_wrong_right(seed, n=50, p=5000, k=100, folds=5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = rng.integers(0, 2, n)
    fs = np.array_split(rng.permutation(n), folds)
    sel = topk_features(X, y, k)
    wrong, right = [], []
    for i in range(folds):
        te = fs[i]
        tr = np.concatenate([fs[j] for j in range(folds) if j != i])
        wrong.append(nearest_centroid_acc(X[tr][:, sel], y[tr], X[te][:, sel], y[te]))
        s2 = topk_features(X[tr], y[tr], k)
        right.append(nearest_centroid_acc(X[tr][:, s2], y[tr], X[te][:, s2], y[te]))
    return float(np.mean(wrong)), float(np.mean(right))


print(cv_wrong_right(0))
