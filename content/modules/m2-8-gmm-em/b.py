import math


def e_step(xs, pis, mus, vars):
    ...


def m_step(xs, R, reg=1e-6):
    ...


def log_likelihood(xs, pis, mus, vars):
    ...


def em(xs, pis, mus, vars, iters):
    ...


R = e_step([1, 2, 4], [0.5, 0.5], [1, 4], [1, 1])
print([round(r[0], 4) for r in R])          # [0.989, 0.8176, 0.011]
print(m_step([1, 2, 4], R))
