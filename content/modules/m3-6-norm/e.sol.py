import torch
import torch.nn.functional as F


def norm_over(x, dims, eps=1e-5):
    mu = x.mean(dim=dims, keepdim=True)
    var = x.var(dim=dims, keepdim=True, unbiased=False)
    return (x - mu) / torch.sqrt(var + eps)


def group_norm(x, G, gamma, beta, eps=1e-5):
    """x: (N, C, H, W). Döngü yok."""
    N, C, H, W = x.shape
    y = norm_over(x.reshape(N, G, C // G, H, W), (2, 3, 4), eps).reshape(N, C, H, W)
    return y * gamma[:, None, None] + beta[:, None, None]


x = torch.randn(2, 6, 4, 4)
print((group_norm(x, 3, torch.ones(6), torch.zeros(6)) - F.group_norm(x, 3)).abs().max())
