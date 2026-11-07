import math
import torch
import torch.nn.functional as F


def gradient_field(x, nbins=9):
    """x (B, H, W) → (M, theta, hist (B, nbins)). Döngü yok."""
    B, H, W = x.shape
    kx = torch.tensor([[-1.0, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype)
    k = torch.stack([kx, kx.T])[:, None]
    g = F.conv2d(F.pad(x[:, None], (1, 1, 1, 1), mode="reflect"), k)
    gx, gy = g[:, 0], g[:, 1]
    M = torch.sqrt(gx * gx + gy * gy)
    theta = torch.remainder(torch.atan2(gy, gx), math.pi)
    b = (theta / math.pi * nbins).long().clamp(max=nbins - 1)
    idx = b + nbins * torch.arange(B)[:, None, None]
    hist = torch.zeros(B * nbins, dtype=x.dtype).scatter_add_(0, idx.reshape(-1), M.reshape(-1)).reshape(B, nbins)
    return M, theta, hist


x = torch.rand(2, 32, 32)
M, th, h = gradient_field(x)
print(M.shape, th.min().item() >= 0, h.shape)
