import math


def cfg(eps_u, eps_c, s):
    return [u + s * (c - u) for u, c in zip(eps_u, eps_c)]


def classifier_guided(eps, abar, s, grad):
    return eps - s * math.sqrt(1 - abar) * grad


def guided_gauss(m0, v0, ml, vl, s):
    prec = 1 / v0 + s / vl
    return (m0 / v0 + s * ml / vl) / prec, 1 / prec


def cosine_abar(t, T, s=0.008):
    f = lambda u: math.cos((u / T + s) / (1 + s) * math.pi / 2) ** 2
    return f(t) / f(0)


def linear_abar(t, T, b1=1e-4, bT=0.02):
    a = 1.0
    for i in range(1, t + 1):
        b = b1 + (bT - b1) * (i - 1) / (T - 1)
        a *= 1 - b
    return a


def snr(abar):
    return abar / (1 - abar)
