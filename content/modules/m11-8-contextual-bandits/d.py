import torch
import torch.nn as nn


def xor_means(x):
    p = (x[:, 0] * x[:, 1] > 0).double()
    return torch.stack([torch.full_like(p, 0.6), p, 1 - p], 1)


class BLR:
    def __init__(self, K, d, lam=1.0, sigma=0.5):
        ...

    def update(self, a, z, r):
        ...

    def refit(self, Z, A, R):
        ...

    def sample(self, z, g):
        ...


def run_lints(T, seed=0):
    ...


def run_neural_linear(T, seed=0, retrain=50, iters=300):
    ...


print(run_lints(300)[-1])
