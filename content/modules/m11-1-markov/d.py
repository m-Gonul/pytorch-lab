import torch


def simulate(T, s0, steps, n, seed=0):
    ...


def count_mle(traj, M):
    ...


def grad_mle(traj, M, steps=500, lr=0.5):
    ...


T = torch.tensor([[0.9, 0.5], [0.1, 0.5]], dtype=torch.float64)
tr = simulate(T, 0, 50, 100)
print(count_mle(tr, 2))
