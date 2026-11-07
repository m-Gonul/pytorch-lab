import torch


def batched_pca(Xs, k):
    """Xs: (B, n, d) → (components (B, k, d), variances (B, k)). Döngü yok."""
    ...


Xs = torch.randn(4, 50, 3) * torch.tensor([3.0, 1.0, 0.3])
P, var = batched_pca(Xs, 2)
print(P.shape, var)
