import math


def covariance(X):
    ...


def power_iteration(C, iters=200):
    ...


def top_k(C, k, iters=200):
    ...


print(power_iteration([[2.0, 1.0], [1.0, 2.0]]))   # (3.0, [0.7071, 0.7071])
print(top_k([[2.0, 1.0], [1.0, 2.0]], 2))
