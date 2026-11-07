import math


def f(p):
    x, y = p
    return x * x - y * y + 0.25 * y ** 4


def grad(p):
    ...


def hess(p):
    ...


def newton(p0, n):
    ...


def trust_region_saddle(p0, n, delta0=0.5):
    ...


print(newton([1.0, 0.01], 20)[-1], trust_region_saddle([1.0, 0.01], 50)[-1])
