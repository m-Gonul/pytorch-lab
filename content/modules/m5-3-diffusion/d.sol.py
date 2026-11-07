import math
import torch
from torch import nn


def make_schedule(T=100, beta1=1e-4, betaT=0.05):
    betas = torch.linspace(beta1, betaT, T)
    return betas, torch.cumprod(1 - betas, 0)


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        half = dim // 2
        self.register_buffer("freqs", torch.exp(-math.log(1000) * torch.arange(half) / half))

    def forward(self, t):
        a = t[:, None].float() * self.freqs
        return torch.cat([a.sin(), a.cos()], dim=1)


class EpsNet(nn.Module):
    def __init__(self, hidden=128, t_dim=32):
        super().__init__()
        self.emb = TimeEmbedding(t_dim)
        self.net = nn.Sequential(nn.Linear(2 + t_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 2))

    def forward(self, x, t):
        return self.net(torch.cat([x, self.emb(t)], dim=1))


def ddpm_loss(model, x0, abar, t=None, eps=None):
    if t is None:
        t = torch.randint(0, len(abar), (x0.shape[0],))
    if eps is None:
        eps = torch.randn_like(x0)
    a = abar[t][:, None]
    xt = a.sqrt() * x0 + (1 - a).sqrt() * eps
    return ((model(xt, t) - eps) ** 2).mean()


@torch.no_grad()
def ddpm_sample(model, betas, abar, n):
    x = torch.randn(n, 2)
    for t in reversed(range(len(betas))):
        e = model(x, torch.full((n,), t, dtype=torch.long))
        x = (x - betas[t] / (1 - abar[t]).sqrt() * e) / (1 - betas[t]).sqrt()
        if t > 0:
            x = x + ((1 - abar[t - 1]) / (1 - abar[t]) * betas[t]).sqrt() * torch.randn_like(x)
    return x


def ring(n):
    k = torch.randint(0, 8, (n,))
    ang = k * 2 * math.pi / 8
    return torch.stack([2 * ang.cos(), 2 * ang.sin()], 1) + 0.05 * torch.randn(n, 2)


torch.manual_seed(0)
betas, abar = make_schedule()
model = EpsNet()
opt = torch.optim.Adam(model.parameters(), lr=2e-3)
for step in range(1000):
    loss = ddpm_loss(model, ring(512), abar)
    opt.zero_grad(); loss.backward(); opt.step()
print(loss.item(), ddpm_sample(model, betas, abar, 5))
