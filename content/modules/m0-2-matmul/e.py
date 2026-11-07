import torch


def batched_matmul(As, Bs):
    """As: (N, n, k), Bs: (N, k, m) → (N, n, m). Döngü yok."""
    ...


As = torch.randn(256, 8, 8)
Bs = torch.randn(256, 8, 8)
print(batched_matmul(As, Bs).shape)  # torch.Size([256, 8, 8])
