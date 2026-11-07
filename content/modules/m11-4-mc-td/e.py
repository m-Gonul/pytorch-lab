import torch


def batched_td(n_runs, n_steps, alpha, n=5, seed=0):
    ...


V, rms = batched_td(100, 500, 0.1)
print(V.shape, rms[-1])
