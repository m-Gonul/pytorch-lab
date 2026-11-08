import numpy as np


def latent_loss(mu, gamma):
    return -0.5 * np.sum(1 + gamma - np.exp(gamma) - mu ** 2, -1)


def elbo(x, a, s2, m, v):
    rec = -0.5 * np.log(2 * np.pi * s2) - ((x - a * m) ** 2 + a ** 2 * v) / (2 * s2)
    return rec - latent_loss(np.atleast_1d(m), np.log(np.atleast_1d(v)))


def log_evidence(x, a, s2):
    var = a ** 2 + s2
    return -0.5 * np.log(2 * np.pi * var) - x ** 2 / (2 * var)


def true_posterior(x, a, s2):
    v = 1 / (1 + a ** 2 / s2)
    return v * a * x / s2, v
