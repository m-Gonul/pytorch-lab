import torch


def wiener(n_paths, n_steps, t_max=1.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    dW = torch.randn(n_paths, n_steps, generator=g, dtype=torch.float64) * (t_max / n_steps) ** 0.5
    return torch.cat([torch.zeros(n_paths, 1, dtype=torch.float64), dW.cumsum(1)], 1)


def quadratic_variation(W):
    return (W.diff(dim=1) ** 2).sum(1)


def max_exceed(W, a):
    return (W.max(1).values >= a).double().mean().item(), 2 * (W[:, -1] >= a).double().mean().item()


W = wiener(5, 100)
print(W.shape)
