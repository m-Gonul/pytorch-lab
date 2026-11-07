import torch


def batched_pca(Xs, k):
    """Xs: (B, n, d) → (components (B, k, d), variances (B, k)). Döngü yok."""
    Xc = Xs - Xs.mean(1, keepdim=True)
    U, s, Vh = torch.linalg.svd(Xc, full_matrices=False)
    V = Vh[:, :k]
    idx = V.abs().argmax(-1, keepdim=True)
    return V * V.gather(-1, idx).sign(), s[:, :k] ** 2 / (Xs.shape[1] - 1)


Xs = torch.randn(4, 50, 3) * torch.tensor([3.0, 1.0, 0.3])
P, var = batched_pca(Xs, 2)
print(P.shape, var)
