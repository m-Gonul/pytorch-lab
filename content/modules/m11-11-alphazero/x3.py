import numpy as np


def elo_diff_mle(wins, losses):
    ...


def elo_posterior(wins, losses, n_samples=200000, seed=0):
    ...


print(elo_diff_mle(4, 1), elo_posterior(4, 1))
