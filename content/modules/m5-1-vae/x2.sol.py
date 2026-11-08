import numpy as np


def active_dims(eigs, sigma2, beta):
    return int(np.sum(np.asarray(eigs) > beta * sigma2))


def decoder_norms(eigs, sigma2, beta):
    return np.sqrt(np.maximum(0, np.asarray(eigs, float) - beta * sigma2))


def kl_per_dim(eigs, sigma2, beta):
    e = np.asarray(eigs, float)
    return np.where(e > beta * sigma2, 0.5 * np.log(np.maximum(e, 1e-300) / (beta * sigma2)), 0.0)
