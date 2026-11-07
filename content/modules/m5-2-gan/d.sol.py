import math
import torch
import torch.nn.functional as F
from torch import nn


class Generator(nn.Module):
    def __init__(self, z_dim=2, hidden=128):
        super().__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(nn.Linear(z_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x)


def d_loss(D, real, fake):
    lr, lf = D(real), D(fake.detach())
    return F.binary_cross_entropy_with_logits(lr, torch.ones_like(lr)) + F.binary_cross_entropy_with_logits(lf, torch.zeros_like(lf))


def g_loss(D, fake):
    lf = D(fake)
    return F.binary_cross_entropy_with_logits(lf, torch.ones_like(lf))


def train_gan(G, D, sample_real, steps=3000, batch=256, lr=1e-3):
    oG = torch.optim.Adam(G.parameters(), lr, betas=(0.5, 0.999))
    oD = torch.optim.Adam(D.parameters(), lr, betas=(0.5, 0.999))
    for _ in range(steps):
        fake = G(torch.randn(batch, G.z_dim))
        lD = d_loss(D, sample_real(batch), fake)
        oD.zero_grad(); lD.backward(); oD.step()
        lG = g_loss(D, fake)
        oG.zero_grad(); lG.backward(); oG.step()
    return G, D


def ring(n):
    k = torch.randint(0, 8, (n,))
    ang = k * 2 * math.pi / 8
    return torch.stack([2 * ang.cos(), 2 * ang.sin()], 1) + 0.05 * torch.randn(n, 2)


torch.manual_seed(0)
G, D = Generator(), Discriminator()
train_gan(G, D, ring, steps=1000)
print(G(torch.randn(5, 2)))
