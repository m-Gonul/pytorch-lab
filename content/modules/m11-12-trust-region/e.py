import torch


def gae_scan(r, v, d, v_last, gamma, lam):
    ...


def gae_matrix(r, v, d, v_last, gamma, lam):
    ...


def walk_batch(N, T, g, n=19):
    ...


g = torch.Generator().manual_seed(0)
S, R, D = walk_batch(4, 50, g)
print(S.shape, R.sum(0))
