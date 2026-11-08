import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

T = 200
BETAS = torch.linspace(1e-4, 0.1, T)
AB = torch.cumprod(1 - BETAS, 0)


class CondDenoiser(nn.Module):
    def __init__(self, D, C=10, h=256):
        super().__init__()
        self.C = C
        self.emb = nn.Embedding(C + 1, h)
        self.t = nn.Linear(16, h)
        self.inp = nn.Linear(D, h)
        self.net = nn.Sequential(nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, D))

    def forward(self, x, t, y):
        f = torch.exp(-math.log(1000) * torch.arange(8) / 8)
        a = t[:, None].float() * f[None]
        return self.net(self.inp(x) + self.t(torch.cat([a.sin(), a.cos()], 1)) + self.emb(y))


def train_diffusion(Z, y, steps, seed):
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    m = CondDenoiser(Z.shape[1])
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    for _ in range(steps):
        idx = torch.randint(0, len(Z), (128,), generator=g)
        x0, yy = Z[idx], y[idx]
        t = torch.randint(0, T, (128,), generator=g)
        e = torch.randn(x0.shape, generator=g)
        yy = torch.where(torch.rand(128, generator=g) < 0.15, torch.full_like(yy, m.C), yy)
        a = AB[t][:, None]
        loss = F.mse_loss(m(a.sqrt() * x0 + (1 - a).sqrt() * e, t, yy), e)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return m


@torch.no_grad()
def sample(model, labels, D, s, seed):
    g = torch.Generator().manual_seed(seed)
    n = len(labels)
    x = torch.randn(n, D, generator=g)
    yu = torch.full((n,), model.C)
    for i in range(T - 1, -1, -1):
        t = torch.full((n,), i)
        eu = model(x, t, yu)
        e = eu + s * (model(x, t, labels) - eu)
        x0 = (x - (1 - AB[i]).sqrt() * e) / AB[i].sqrt()
        x = AB[i - 1].sqrt() * x0 + (1 - AB[i - 1]).sqrt() * e if i > 0 else x0
    return x
