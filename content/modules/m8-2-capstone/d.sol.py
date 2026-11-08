import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits

T = 200
BETAS = torch.linspace(1e-4, 0.05, T)
AB = torch.cumprod(1 - BETAS, 0)


def digits16():
    X, y = load_digits(return_X_y=True)
    X = torch.tensor(X, dtype=torch.float32).view(-1, 1, 8, 8) / 16
    return F.interpolate(X, size=16, mode="bilinear", align_corners=False) * 2 - 1, torch.tensor(y)


def temb(t, d):
    f = torch.exp(-math.log(10000) * torch.arange(d // 2) / (d // 2))
    a = t[:, None].float() * f[None]
    return torch.cat([a.sin(), a.cos()], 1)


class ResBlock(nn.Module):
    def __init__(self, ci, co, e):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(4, ci), nn.Conv2d(ci, co, 3, padding=1)
        self.t = nn.Linear(e, co)
        self.n2, self.c2 = nn.GroupNorm(4, co), nn.Conv2d(co, co, 3, padding=1)
        nn.init.zeros_(self.c2.weight)
        nn.init.zeros_(self.c2.bias)
        self.skip = nn.Conv2d(ci, co, 1) if ci != co else nn.Identity()

    def forward(self, x, e):
        h = self.c1(F.silu(self.n1(x))) + self.t(e)[:, :, None, None]
        return self.skip(x) + self.c2(F.silu(self.n2(h)))


class Attn(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n, self.qkv, self.o = nn.GroupNorm(4, c), nn.Conv2d(c, 3 * c, 1), nn.Conv2d(c, c, 1)
        nn.init.zeros_(self.o.weight)
        nn.init.zeros_(self.o.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.n(x)).view(B, 3, C, H * W).unbind(1)
        a = torch.softmax(q.transpose(1, 2) @ k / math.sqrt(C), -1)
        return x + self.o((v @ a.transpose(1, 2)).view(B, C, H, W))


class UNet(nn.Module):
    def __init__(self, c=16, C=10):
        super().__init__()
        e = 4 * c
        self.c, self.C = c, C
        self.temb = nn.Sequential(nn.Linear(c, e), nn.SiLU(), nn.Linear(e, e))
        self.cemb = nn.Embedding(C + 1, e)
        self.inp = nn.Conv2d(1, c, 3, padding=1)
        self.d1, self.down1 = ResBlock(c, c, e), nn.Conv2d(c, c, 3, 2, 1)
        self.d2, self.down2 = ResBlock(c, 2 * c, e), nn.Conv2d(2 * c, 2 * c, 3, 2, 1)
        self.m1, self.att, self.m2 = ResBlock(2 * c, 2 * c, e), Attn(2 * c), ResBlock(2 * c, 2 * c, e)
        self.u2, self.u1 = ResBlock(4 * c, 2 * c, e), ResBlock(3 * c, c, e)
        self.out = nn.Sequential(nn.GroupNorm(4, c), nn.SiLU(), nn.Conv2d(c, 1, 3, padding=1))

    def forward(self, x, t, y):
        e = self.temb(temb(t, self.c)) + self.cemb(y)
        h1 = self.d1(self.inp(x), e)
        h2 = self.d2(self.down1(h1), e)
        h = self.m2(self.att(self.m1(self.down2(h2), e)), e)
        h = self.u2(torch.cat([F.interpolate(h, scale_factor=2), h2], 1), e)
        h = self.u1(torch.cat([F.interpolate(h, scale_factor=2), h1], 1), e)
        return self.out(h)


def train(model, steps, seed, ema_decay=0.99, p_uncond=0.1):
    X, y = digits16()
    g = torch.Generator().manual_seed(seed)
    ema = copy.deepcopy(model)
    opt = torch.optim.AdamW(model.parameters(), 2e-3)
    for _ in range(steps):
        idx = torch.randint(0, len(X), (64,), generator=g)
        x0, yy = X[idx], y[idx]
        t = torch.randint(0, T, (64,), generator=g)
        eps = torch.randn(x0.shape, generator=g)
        yy = torch.where(torch.rand(64, generator=g) < p_uncond, torch.full_like(yy, model.C), yy)
        a = AB[t][:, None, None, None]
        loss = F.mse_loss(model(a.sqrt() * x0 + (1 - a).sqrt() * eps, t, yy), eps)
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - ema_decay)
    return model, ema


@torch.no_grad()
def sample(net, labels, s, steps, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(len(labels), 1, 16, 16, generator=g)
    ts = torch.linspace(T - 1, 0, steps).round().long()
    yu = torch.full_like(labels, net.C)
    for k, i in enumerate(ts):
        tt = torch.full((len(labels),), int(i))
        eu = net(x, tt, yu)
        e = eu + s * (net(x, tt, labels) - eu)
        x0 = ((x - (1 - AB[i]).sqrt() * e) / AB[i].sqrt()).clamp(-1, 1)
        ap = AB[ts[k + 1]] if k + 1 < len(ts) else torch.tensor(1.0)
        x = ap.sqrt() * x0 + (1 - ap).sqrt() * e
    return x
