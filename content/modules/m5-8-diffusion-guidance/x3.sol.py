import numpy as np


def eps_to_x0(xt, eps, abar):
    return (xt - np.sqrt(1 - abar) * eps) / np.sqrt(abar)


def x0_to_eps(xt, x0, abar):
    return (xt - np.sqrt(abar) * x0) / np.sqrt(1 - abar)


def eps_to_v(xt, eps, abar):
    a, s = np.sqrt(abar), np.sqrt(1 - abar)
    return a * eps - s * eps_to_x0(xt, eps, abar)


def v_to_eps(xt, v, abar):
    a, s = np.sqrt(abar), np.sqrt(1 - abar)
    return s * xt + a * v


def eps_to_score(eps, abar):
    return -eps / np.sqrt(1 - abar)
