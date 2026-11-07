import math
import torch
import torch.nn.functional as F
from torch import nn


class Generator(nn.Module):
    def __init__(self, z_dim=2, hidden=128):
        super().__init__()
        ...

    def forward(self, z):
        ...


class Discriminator(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        ...

    def forward(self, x):
        ...


def d_loss(D, real, fake):
    ...


def g_loss(D, fake):
    ...


def train_gan(G, D, sample_real, steps=3000, batch=256, lr=1e-3):
    ...


def ring(n):
    k = torch.randint(0, 8, (n,))
    ang = k * 2 * math.pi / 8
    return torch.stack([2 * ang.cos(), 2 * ang.sin()], 1) + 0.05 * torch.randn(n, 2)


torch.manual_seed(0)
G, D = Generator(), Discriminator()
train_gan(G, D, ring, steps=1000)
print(G(torch.randn(5, 2)))
