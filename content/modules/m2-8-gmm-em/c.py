import numpy as np


def log_gauss(X, mus, covs):
    """→ (n, K). Döngü yok."""
    ...


def e_step_np(X, pis, mus, covs):
    ...


def m_step_np(X, R, reg=1e-6):
    ...


def em_np(X, pis, mus, covs, iters):
    ...


rng = np.random.default_rng(0)
X = np.concatenate([rng.multivariate_normal([0, 0], [[1, 0.8], [0.8, 1]], 150), rng.normal([4, 0], 0.5, (100, 2))])
pis, mus, covs, lls = em_np(X, np.array([0.5, 0.5]), X[[0, -1]], np.stack([np.eye(2)] * 2), 30)
print(mus, lls[-1])
