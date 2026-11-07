import numpy as np


def posterior(X, r, lam, sigma):
    ...


def sequential_posterior(X, r, lam, sigma):
    ...


def predictive(Xq, mu, Sigma, sigma):
    ...


def joint_features(xs, XA):
    ...


X = np.array([[2.0]]); r = np.array([3.0])
print(posterior(X, r, 1.0, 1.0))
