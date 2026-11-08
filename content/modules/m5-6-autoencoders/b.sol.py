import math
import random


def ae_params(sizes, tied=False):
    enc = sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:]))
    dec_b = sum(sizes[:-1])
    dec_w = sum(a * b for a, b in zip(sizes[:-1], sizes[1:]))
    return enc + dec_b + (0 if tied else dec_w)


def kl_sparsity(rho, rho_hat):
    return rho * math.log(rho / rho_hat) + (1 - rho) * math.log((1 - rho) / (1 - rho_hat))


def kl_sparsity_grad(rho, rho_hat):
    return -rho / rho_hat + (1 - rho) / (1 - rho_hat)


def greedy_plan(sizes):
    return list(zip(sizes[:-1], sizes[1:]))


def masking_noise(x, p, rng):
    return [0.0 if rng.random() < p else v for v in x]


def gaussian_noise(x, sigma, rng):
    return [v + rng.gauss(0, sigma) for v in x]
