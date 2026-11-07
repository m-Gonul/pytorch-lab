import torch
from torch.distributions import Normal


def normal_nll(x, mu, sigma):
    """Ortalama negatif log-likelihood."""
    return -Normal(mu, sigma).log_prob(x).mean()


def fit_normal(x, steps=2000, lr=0.05):
    """σ = exp(s) ile Adam → (mu, sigma) skaler tensörler."""
    mu = torch.zeros((), requires_grad=True)
    s = torch.zeros((), requires_grad=True)
    opt = torch.optim.Adam([mu, s], lr=lr)
    for _ in range(steps):
        loss = normal_nll(x, mu, s.exp())
        opt.zero_grad()
        loss.backward()
        opt.step()
    return mu.detach(), s.exp().detach()


x = torch.tensor([2.0, 4.0, 6.0])
print(normal_nll(x, torch.tensor(4.0), torch.tensor(1.0)))
print(fit_normal(x))     # ≈ (4.0, 1.633)
