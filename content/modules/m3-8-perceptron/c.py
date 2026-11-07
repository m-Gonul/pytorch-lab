import numpy as np


def relu_features(X, W):
    ...


def min_norm_fit(F, y):
    ...


def double_descent(n=100, d=10, ps=(10, 25, 50, 75, 90, 100, 110, 125, 150, 200, 400, 1000, 3000), trials=20, sigma=0.3, seed=0):
    ...


def relu_interpolant(knots, values):
    ...


print(relu_interpolant(np.array([0.0, 1, 2, 3]), np.array([0.0, 1, 0, 2])))
