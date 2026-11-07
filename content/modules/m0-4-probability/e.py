import torch


def bootstrap_means(x, B, generator):
    """→ (B,) bootstrap ortalamaları. Döngü yok."""
    ...


def batched_gaussian_mle(X):
    """X: (B, n) → (mu: (B,), var: (B,)). Döngü yok."""
    ...


g = torch.Generator().manual_seed(0)
x = torch.randn(100, generator=g) * 2 + 5
m = bootstrap_means(x, 10000, g)
print(m.mean(), m.std(), x.std(unbiased=False) / 10)
