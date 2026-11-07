import torch


def q_value_iteration(P, R, gamma, tol=1e-10):
    ...


def sync_q_learning(P, R, gamma, n_iter, rate="rescaled", seed=0):
    ...


def greedy(Q):
    ...


g = torch.Generator().manual_seed(0)
P = torch.rand(3, 8, 8, generator=g, dtype=torch.float64); P /= P.sum(-1, keepdim=True)
R = torch.rand(8, 3, generator=g, dtype=torch.float64)
print((sync_q_learning(P, R, 0.9, 1000) - q_value_iteration(P, R, 0.9)).abs().max())
