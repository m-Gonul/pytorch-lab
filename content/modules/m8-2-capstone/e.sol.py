import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

T = 200
BETAS = torch.linspace(1e-4, 0.05, T)
AB = torch.cumprod(1 - BETAS, 0)


def cfg_two_calls(net, x, t, y, s):
    eu = net(x, t, torch.full_like(y, net.C))
    return eu + s * (net(x, t, y) - eu)


def cfg_batched(net, x, t, y, s):
    out = net(torch.cat([x, x]), torch.cat([t, t]), torch.cat([torch.full_like(y, net.C), y]))
    eu, ec = out.chunk(2)
    return eu + s * (ec - eu)


@torch.no_grad()
def ddpm_sample(net, labels, s, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(len(labels), 1, 16, 16, generator=g)
    for i in range(T - 1, -1, -1):
        tt = torch.full((len(labels),), i)
        e = cfg_batched(net, x, tt, labels, s)
        x0 = ((x - (1 - AB[i]).sqrt() * e) / AB[i].sqrt()).clamp(-1, 1)
        if i == 0:
            return x0
        abp = AB[i - 1]
        mean = abp.sqrt() * BETAS[i] / (1 - AB[i]) * x0 + (1 - BETAS[i]).sqrt() * (1 - abp) / (1 - AB[i]) * x
        var = (1 - abp) / (1 - AB[i]) * BETAS[i]
        x = mean + var.sqrt() * torch.randn(x.shape, generator=g)


@torch.no_grad()
def ddim_sample(net, labels, s, steps, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(len(labels), 1, 16, 16, generator=g)
    ts = torch.linspace(T - 1, 0, steps).round().long()
    for k, i in enumerate(ts):
        tt = torch.full((len(labels),), int(i))
        e = cfg_batched(net, x, tt, labels, s)
        x0 = ((x - (1 - AB[i]).sqrt() * e) / AB[i].sqrt()).clamp(-1, 1)
        ap = AB[ts[k + 1]] if k + 1 < len(ts) else torch.tensor(1.0)
        x = ap.sqrt() * x0 + (1 - ap).sqrt() * e
    return x
