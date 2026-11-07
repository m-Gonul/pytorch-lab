import numpy as np


def sigmoid_np(z):
    ...


def fit_logreg(X, y, lr=0.5, steps=500):
    """→ (w, b, losses). Yalnız adım döngüsü."""
    ...


def predict(X, w, b):
    ...


rng = np.random.default_rng(0)
X = np.concatenate([rng.normal(-1.5, 1, (100, 2)), rng.normal(1.5, 1, (100, 2))])
y = np.repeat([0, 1], 100)
w, b, losses = fit_logreg(X, y)
print("doğruluk:", (predict(X, w, b) == y).mean(), "son kayıp:", losses[-1])
