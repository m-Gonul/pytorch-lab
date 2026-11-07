import numpy as np


def posterior(X, r, lam, sigma):
    d = X.shape[1]
    Sigma = np.linalg.inv(X.T @ X / sigma ** 2 + np.eye(d) / lam ** 2)
    return Sigma @ X.T @ r / sigma ** 2, Sigma


def sequential_posterior(X, r, lam, sigma):
    d = X.shape[1]
    Sigma = np.eye(d) * lam ** 2
    b = np.zeros(d)
    for x, y in zip(X, r):
        Sx = Sigma @ x
        Sigma = Sigma - np.outer(Sx, Sx) / (sigma ** 2 + x @ Sx)
        b = b + x * y / sigma ** 2
    return Sigma @ b, Sigma


def predictive(Xq, mu, Sigma, sigma):
    return Xq @ mu, np.einsum("ni,ij,nj->n", Xq, Sigma, Xq) + sigma ** 2


def joint_features(xs, XA):
    return np.einsum("i,kj->kij", xs, XA).reshape(XA.shape[0], -1)


X = np.array([[2.0]]); r = np.array([3.0])
print(posterior(X, r, 1.0, 1.0))
