import torch


def bagged_stumps(X, y, B, generator):
    """→ (j, t, sol, sağ), her biri (B,). Döngü yok."""
    ...


def bagged_predict(model, X):
    """→ (n,) B stump'ın ortalaması. Döngü yok."""
    ...


g = torch.Generator().manual_seed(0)
X = torch.rand(200, 3, generator=g)
y = torch.sin(6 * X[:, 0]) + 0.1 * torch.randn(200, generator=g)
model = bagged_stumps(X, y, 100, g)
print(((bagged_predict(model, X) - y) ** 2).mean())
