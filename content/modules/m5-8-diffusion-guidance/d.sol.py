import math
import torch
import torch.nn as nn
import torch.nn.functional as F

ANG = torch.arange(8) * math.pi / 4
MU = 4 * torch.stack([torch.cos(ANG), torch.sin(ANG)], 1)
LAB = torch.arange(8) % 4
T = 200
BETAS = torch.linspace(1e-4, 0.1, T)
AB = torch.cumprod(1 - BETAS, 0)


def data(n, g):
    k = torch.randint(0, 8, (n,), generator=g)
    return MU[k] + 0.5 * torch.randn(n, 2, generator=g), LAB[k]


class EpsNet(nn.Module):
    def __init__(self, C=4, h=128):
        super().__init__()
        self.C = C
        self.emb = nn.Embedding(C + 1, h)
        self.t = nn.Linear(16, h)
        self.inp = nn.Linear(2, h)
        self.net = nn.Sequential(nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 2))

    def forward(self, x, t, y):
        f = torch.exp(-math.log(1000) * torch.arange(8) / 8)
        a = t[:, None].float() * f[None]
        return self.net(self.inp(x) + self.t(torch.cat([a.sin(), a.cos()], 1)) + self.emb(y))


def train(model, steps, p_uncond=0.15, seed=0):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), 2e-3)
    for _ in range(steps):
        x0, y = data(256, g)
        t = torch.randint(0, T, (256,), generator=g)
        e = torch.randn(256, 2, generator=g)
        y = torch.where(torch.rand(256, generator=g) < p_uncond, torch.full_like(y, model.C), y)
        a = AB[t][:, None]
        loss = F.mse_loss(model(a.sqrt() * x0 + (1 - a).sqrt() * e, t, y), e)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def sample(model, n, y, s, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 2, generator=g)
    yc, yu = torch.full((n,), y), torch.full((n,), model.C)
    for i in range(T - 1, -1, -1):
        t = torch.full((n,), i)
        eu = model(x, t, yu)
        e = eu + s * (model(x, t, yc) - eu)
        x0 = (x - (1 - AB[i]).sqrt() * e) / AB[i].sqrt()
        x = AB[i - 1].sqrt() * x0 + (1 - AB[i - 1]).sqrt() * e if i > 0 else x0
    return x
