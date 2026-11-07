import torch


def normal_posterior(sum_r, n, mu0, var0, noise_var):
    ...


def thompson(probs, T, n_runs, k=1, seed=0):
    ...


def gaussian_thompson(means, T, n_runs, noise_sd=1.0, seed=0):
    ...


print(thompson([0.9, 0.8, 0.5], 200, 100)[-1])
