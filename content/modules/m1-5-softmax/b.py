import math


def logsumexp(z):
    ...


def softmax(z):
    ...


def cross_entropy(z, y):
    ...


def grad_z(z, y):
    ...


print(softmax([2, 1, 0]))                  # [0.665, 0.245, 0.090]
print(cross_entropy([2, 1, 0], 0))         # 0.4076
print(softmax([1000, 999, 998]))           # aynı, taşma yok
