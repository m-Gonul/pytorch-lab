import torch
from torch.distributions import Normal


def normal_nll(x, mu, sigma):
    """Ortalama negatif log-likelihood."""
    ...


def fit_normal(x, steps=2000, lr=0.05):
    """σ = exp(s) ile Adam → (mu, sigma) skaler tensörler."""
    mu = torch.zeros((), requires_grad=True)
    s = torch.zeros((), requires_grad=True)
    ...


x = torch.tensor([2.0, 4.0, 6.0])
print(normal_nll(x, torch.tensor(4.0), torch.tensor(1.0)))
print(fit_normal(x))     # ≈ (4.0, 1.633)
