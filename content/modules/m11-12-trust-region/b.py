import math


def rosen(p):
    ...


def rosen_grad(p):
    ...


def rosen_hess(p):
    ...


def gd_backtracking(p0, n_iter):
    ...


def dogleg(g, H, delta):
    ...


def trust_region(p0, n_iter, delta0=1.0, delta_max=10.0, eta=0.1):
    ...


print(trust_region([-1.2, 1.0], 50)[-1], gd_backtracking([-1.2, 1.0], 50)[-1])
