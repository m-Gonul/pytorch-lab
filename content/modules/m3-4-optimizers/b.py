import math


def sgd_momentum(w, g, state, lr, beta=0.9):
    ...


def rmsprop(w, g, state, lr=0.01, alpha=0.99, eps=1e-8):
    ...


def adam(w, g, state, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.0, decoupled=False):
    ...


w, st = [0.0], {}
for _ in range(2):
    w = adam(w, [2 * (w[0] - 3)], st, lr=0.1, eps=0.0)
print(w)       # [0.19989…]
