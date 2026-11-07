import numpy as np


def ridge(X, y, lam):
    """→ (w, b); bias cezalandırılmaz. Döngü yok."""
    ...


def ridge_path(X, y, lams):
    """→ (L, d): her λ için w; tek SVD. Döngü yok."""
    ...


rng = np.random.default_rng(0)
X = rng.normal(size=(50, 5))
y = X @ np.array([1.0, 0, -2, 0, 0.5]) + 3 + 0.1 * rng.normal(size=50)
print(ridge(X, y, 1.0))
print(ridge_path(X, y, np.array([0.0, 1.0, 100.0])))
