import numpy as np


def sqdist(X, C):
    return np.maximum((X ** 2).sum(1)[:, None] + (C ** 2).sum(1)[None] - 2 * X @ C.T, 0)


def kmeans_np(X, C0, iters):
    """→ (C, labels, inertia). Yalnız iterasyon döngüsü."""
    C = np.asarray(C0, float).copy()
    k = len(C)
    for _ in range(iters):
        labels = sqdist(X, C).argmin(1)
        A = np.eye(k)[labels]
        counts = A.sum(0)
        sums = A.T @ X
        C = np.where(counts[:, None] > 0, sums / np.maximum(counts, 1)[:, None], C)
    D = sqdist(X, C)
    labels = D.argmin(1)
    return C, labels, float(D[np.arange(len(X)), labels].sum())


def kmeanspp(X, k, rng):
    """→ (k, d). rng çağrı sırası: integers(n), sonra her merkez için choice(n, p=…)."""
    n = len(X)
    C = [X[rng.integers(n)]]
    for _ in range(k - 1):
        D2 = sqdist(X, np.array(C)).min(1)
        C.append(X[rng.choice(n, p=D2 / D2.sum())])
    return np.array(C)


rng = np.random.default_rng(0)
X = np.concatenate([rng.normal(c, 0.5, (50, 2)) for c in ([0, 0], [5, 5], [0, 5])])
C0 = kmeanspp(X, 3, np.random.default_rng(1))
print(kmeans_np(X, C0, 10)[2])
