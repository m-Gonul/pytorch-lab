import torch


def bootstrap_means(x, B, generator):
    """→ (B,) bootstrap ortalamaları. Döngü yok."""
    n = x.shape[0]
    idx = torch.randint(n, (B, n), generator=generator)
    return x[idx].mean(dim=1)


def batched_gaussian_mle(X):
    """X: (B, n) → (mu: (B,), var: (B,)). Döngü yok."""
    mu = X.mean(dim=1)
    return mu, ((X - mu[:, None]) ** 2).mean(dim=1)


g = torch.Generator().manual_seed(0)
x = torch.randn(100, generator=g) * 2 + 5
m = bootstrap_means(x, 10000, g)
print(m.mean(), m.std(), x.std(unbiased=False) / 10)
