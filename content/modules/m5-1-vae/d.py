import torch
import torch.nn.functional as F
from torch import nn


class VAE(nn.Module):
    def __init__(self, d_in, d_lat, hidden=64):
        super().__init__()
        ...

    def encode(self, x):
        ...

    def decode(self, z):
        ...

    def forward(self, x):
        ...


def elbo_loss(x, xhat, mu, logvar, beta=1.0):
    ...


@torch.no_grad()
def sample(model, n):
    ...


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
