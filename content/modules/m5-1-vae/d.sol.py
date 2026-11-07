import torch
import torch.nn.functional as F
from torch import nn


class VAE(nn.Module):
    def __init__(self, d_in, d_lat, hidden=64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, 2 * d_lat))
        self.dec = nn.Sequential(nn.Linear(d_lat, hidden), nn.ReLU(), nn.Linear(hidden, d_in))
        self.d_lat = d_lat

    def encode(self, x):
        mu, logvar = self.enc(x).chunk(2, dim=-1)
        return mu, logvar

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.decode(z), mu, logvar


def elbo_loss(x, xhat, mu, logvar, beta=1.0):
    rec = ((x - xhat) ** 2).sum(-1)
    kl = 0.5 * (mu ** 2 + logvar.exp() - 1 - logvar).sum(-1)
    return (rec + beta * kl).mean()


@torch.no_grad()
def sample(model, n):
    return model.decode(torch.randn(n, model.d_lat))


g = torch.Generator().manual_seed(0)
A = torch.randn(2, 12, generator=g)
X = torch.tanh(torch.randn(1000, 2, generator=g) @ A) + 0.05 * torch.randn(1000, 12, generator=g)
torch.manual_seed(0)
m = VAE(12, 2)
opt = torch.optim.Adam(m.parameters(), lr=1e-2)
for step in range(1000):
    loss = elbo_loss(X, *m(X))
    opt.zero_grad(); loss.backward(); opt.step()
print(loss.item(), sample(m, 3).shape)
