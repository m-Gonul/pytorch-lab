import math


def sigmoid(z):
    ...


def bce(p, y, eps=1e-12):
    ...


def grad(X, y, w, b):
    """→ (gw: liste, gb, ortalama kayıp)"""
    ...


def fit(X, y, lr=0.5, steps=1000):
    ...


X = [[-1.0], [0.0], [2.0]]
y = [0, 1, 1]
print(grad(X, y, [1.0], 0.0))      # ([-0.169…], -0.1167…, 0.3777…)
print(sigmoid(-1000), sigmoid(1000))
