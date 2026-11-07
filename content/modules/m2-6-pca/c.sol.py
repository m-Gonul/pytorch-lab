import numpy as np


def pca(X, k):
    """→ (components (k, d), variances (k,), Z (n, k)). Döngü yok."""
    X = np.asarray(X, float)
    Xc = X - X.mean(0)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt[:k]
    idx = np.argmax(np.abs(V), axis=1)
    V = V * np.sign(V[np.arange(k), idx])[:, None]
    return V, s[:k] ** 2 / (len(X) - 1), Xc @ V.T


def reconstruct(Z, components, mean):
    return Z @ components + mean


rng = np.random.default_rng(0)
X = rng.normal(size=(200, 3)) @ np.array([[3.0, 1.0, 0], [0, 1.0, 0], [0, 0, 0.1]]) + 5
comps, var, Z = pca(X, 2)
print(comps, var)
print(np.abs(reconstruct(Z, comps, X.mean(0)) - X).max())
