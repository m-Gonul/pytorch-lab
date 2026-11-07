import math
import torch


def vec_bandit(probs, T, n_runs, method, eps=0.1, seed=0):
    ...


def lai_robbins(probs):
    ...


print(vec_bandit([0.9, 0.8, 0.5], 500, 1000, "ucb")[-1])
