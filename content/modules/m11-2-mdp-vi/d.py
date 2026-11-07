import torch


def batched_vi(P, R, gamma, iters):
    ...


def contraction_ratio(P, R, gamma, n=1000, seed=0):
    ...


def soft_v_star(P, R, gamma, iters):
    ...


P = torch.rand(4, 2, 5, 5, dtype=torch.float64); P /= P.sum(-1, keepdim=True)
print(batched_vi(P, torch.rand(4, 5, 2, dtype=torch.float64), 0.9, 50)[0].shape)
