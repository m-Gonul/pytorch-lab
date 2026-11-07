import numpy as np


def best_reg_stump(X, r):
    """→ (j, t, sol, sağ) ya da None. Döngü yok."""
    ...


def gbm_np(X, y, rounds, lr):
    ...


def gbm_predict_np(model, X):
    ...


rng = np.random.default_rng(0)
X = rng.uniform(-2, 2, size=(400, 2))
y = np.sin(2 * X[:, 0]) + X[:, 1] ** 2
m = gbm_np(X, y, 100, 0.2)
print(np.mean((gbm_predict_np(m, X) - y) ** 2))
