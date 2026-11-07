import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC


def make_rings(n, seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    r = np.where(y == 1, 1.0, 0.4)
    th = rng.uniform(0, 2 * np.pi, n)
    X = np.stack([r * np.cos(th), r * np.sin(th)], 1) + 0.05 * rng.normal(size=(n, 2))
    return X, y


def lift(X):
    return np.c_[X, (X ** 2).sum(1)]


def _margin(w, b, X, y):
    return float(((2 * y - 1) * (X @ w + b)).min() / np.linalg.norm(w))


def fit_both(X3, y):
    p = Perceptron(max_iter=1000, tol=None, random_state=0).fit(X3, y)
    s = SVC(kernel="linear", C=1e3).fit(X3, y)
    return ((p.score(X3, y), _margin(p.coef_[0], p.intercept_[0], X3, y)), (s.score(X3, y), _margin(s.coef_[0], s.intercept_[0], X3, y)))


X, y = make_rings(200, 0)
print(fit_both(lift(X), y))
