import numpy as np


def pca(X, k):
    """→ (components (k, d), variances (k,), Z (n, k)). Döngü yok."""
    ...


def reconstruct(Z, components, mean):
    ...


rng = np.random.default_rng(0)
X = rng.normal(size=(200, 3)) @ np.array([[3.0, 1.0, 0], [0, 1.0, 0], [0, 0, 0.1]]) + 5
comps, var, Z = pca(X, 2)
print(comps, var)
print(np.abs(reconstruct(Z, comps, X.mean(0)) - X).max())
