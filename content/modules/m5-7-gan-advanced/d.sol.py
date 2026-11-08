import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits


def digits16():
    X, y = load_digits(return_X_y=True)
    X = torch.tensor(X, dtype=torch.float32).view(-1, 1, 8, 8) / 16
    return F.interpolate(X, size=16, mode="bilinear", align_corners=False) * 2 - 1, torch.tensor(y)


class Generator(nn.Module):
    def __init__(self, z=32, c=64):
        super().__init__()
        self.net = nn.Sequential(nn.ConvTranspose2d(z, 2 * c, 4, 1, 0, bias=False), nn.BatchNorm2d(2 * c), nn.ReLU(),
                                 nn.ConvTranspose2d(2 * c, c, 4, 2, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU(),
                                 nn.ConvTranspose2d(c, 1, 4, 2, 1), nn.Tanh())

    def forward(self, z):
        return self.net(z[:, :, None, None])


class Discriminator(nn.Module):
    def __init__(self, c=64):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, c, 4, 2, 1), nn.LeakyReLU(0.2), nn.Conv2d(c, 2 * c, 4, 2, 1, bias=False), nn.BatchNorm2d(2 * c), nn.LeakyReLU(0.2),
                                 nn.Conv2d(2 * c, 1, 4, 1, 0))

    def forward(self, x):
        return self.net(x).view(-1)


def train_gan(G, D, X, epochs, z=32, lr=2e-4):
    og = torch.optim.Adam(G.parameters(), lr, betas=(0.5, 0.999))
    od = torch.optim.Adam(D.parameters(), lr, betas=(0.5, 0.999))
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 64):
            xr = X[perm[i:i + 64]]
            xf = G(torch.randn(len(xr), z))
            ld = F.binary_cross_entropy_with_logits(D(xr), torch.ones(len(xr))) + F.binary_cross_entropy_with_logits(D(xf.detach()), torch.zeros(len(xr)))
            od.zero_grad()
            ld.backward()
            od.step()
            lg = F.binary_cross_entropy_with_logits(D(xf), torch.ones(len(xr)))
            og.zero_grad()
            lg.backward()
            og.step()
    return G


def fid_np(F1, F2):
    F1, F2 = np.asarray(F1, float), np.asarray(F2, float)
    m1, m2 = F1.mean(0), F2.mean(0)
    S1, S2 = np.cov(F1, rowvar=False), np.cov(F2, rowvar=False)
    w, V = np.linalg.eigh(S1)
    r = (V * np.sqrt(np.clip(w, 0, None))) @ V.T
    tr = np.sqrt(np.clip(np.linalg.eigvalsh(r @ S2 @ r), 0, None)).sum()
    return float(((m1 - m2) ** 2).sum() + np.trace(S1) + np.trace(S2) - 2 * tr)
