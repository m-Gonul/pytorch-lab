import math
import torch
from torch import nn


def make_schedule(T=100, beta1=1e-4, betaT=0.05):
    ...


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        ...

    def forward(self, t):
        ...


class EpsNet(nn.Module):
    def __init__(self, hidden=128, t_dim=32):
        super().__init__()
        ...

    def forward(self, x, t):
        ...


def ddpm_loss(model, x0, abar, t=None, eps=None):
    ...


@torch.no_grad()
def ddpm_sample(model, betas, abar, n):
    ...


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
