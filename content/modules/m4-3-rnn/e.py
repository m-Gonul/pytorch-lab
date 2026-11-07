import torch


def linear_scan(a, b):
    """h_t = a_t·h_{t−1} + b_t, h_0 = 0. a, b: (T, …). Yalnız offset döngüsü."""
    ...


a = torch.full((8,), 0.5)
b = torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0])
print(linear_scan(a, b))     # [1, 0.5, 0.25, …]
