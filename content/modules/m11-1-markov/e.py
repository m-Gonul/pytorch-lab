import torch


def wiener(n_paths, n_steps, t_max=1.0, seed=0):
    ...


def quadratic_variation(W):
    ...


def max_exceed(W, a):
    ...


W = wiener(5, 100)
print(W.shape)
