import numpy as np


def ridge(X, y, lam):
    """→ (w, b); bias cezalandırılmaz. Döngü yok."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    xm, ym = X.mean(0), y.mean()
    Xc, yc = X - xm, y - ym
    w = np.linalg.solve(Xc.T @ Xc + lam * np.eye(X.shape[1]), Xc.T @ yc)
    return w, float(ym - xm @ w)


def ridge_path(X, y, lams):
    """→ (L, d): her λ için w; tek SVD. Döngü yok."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    Xc, yc = X - X.mean(0), y - y.mean()
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    D = s / (s ** 2 + np.asarray(lams, float)[:, None])     # (L, k)
    return (D * (U.T @ yc)) @ Vt                            # (L, k) @ (k, d)


rng = np.random.default_rng(0)
X = rng.normal(size=(50, 5))
y = X @ np.array([1.0, 0, -2, 0, 0.5]) + 3 + 0.1 * rng.normal(size=50)
print(ridge(X, y, 1.0))
print(ridge_path(X, y, np.array([0.0, 1.0, 100.0])))
