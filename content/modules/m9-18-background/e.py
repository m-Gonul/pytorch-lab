import torch


def ema_scan(frames, alpha, L=64):
    ...


def ema_var_scan(frames, alpha, L=64, var0=100.0):
    ...


def masks(frames, alpha, k=3.0):
    ...


x = torch.rand(10, 4, 4, dtype=torch.float64)
print(ema_scan(x, 0.1, L=4).shape)
