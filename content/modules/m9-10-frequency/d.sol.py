import numpy as np
import torch


def gaussian_otf(N, s):
    u = torch.fft.fftfreq(N) * N
    v = torch.fft.rfftfreq(N) * N
    return torch.exp(-2 * torch.pi ** 2 * s ** 2 * (u[:, None] ** 2 + v[None, :] ** 2) / N ** 2)


def degrade(x, H, sigma, g):
    y = torch.fft.irfft2(torch.fft.rfft2(x) * H, s=x.shape[-2:])
    return y + sigma * torch.randn(y.shape, generator=g)


def apply(G, y):
    return torch.fft.irfft2(torch.fft.rfft2(y) * G, s=y.shape[-2:])


def wiener(H, K):
    return H / (H * H + K)


def learn_filter(train, H, sigma, steps=600, batch=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    G = torch.ones_like(H, requires_grad=True)
    opt = torch.optim.Adam([G], lr=1e-2)
    for _ in range(steps):
        x = train[torch.randint(0, len(train), (batch,), generator=g)]
        loss = ((apply(G, degrade(x, H, sigma, g)) - x) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return G.detach()


print(gaussian_otf(8, 1.0).shape)
