import numpy as np


def kmeans_np(X, C0, iters):
    """→ (C, labels, inertia). Yalnız iterasyon döngüsü."""
    ...


def kmeanspp(X, k, rng):
    """→ (k, d). rng çağrı sırası: integers(n), sonra her merkez için choice(n, p=…)."""
    ...


rng = np.random.default_rng(0)
X = np.concatenate([rng.normal(c, 0.5, (50, 2)) for c in ([0, 0], [5, 5], [0, 5])])
C0 = kmeanspp(X, 3, np.random.default_rng(1))
print(kmeans_np(X, C0, 10)[2])
