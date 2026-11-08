import torch
import torch.nn as nn
import torch.nn.functional as F


class MappingNet(nn.Module):
    def __init__(self, z=64, w=64, L=8):
        super().__init__()
        layers = []
        for i in range(L):
            layers += [nn.Linear(z if i == 0 else w, w), nn.LeakyReLU(0.2)]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        z = z / torch.sqrt((z ** 2).mean(1, keepdim=True) + 1e-8)
        return self.net(z)


class AdaIN(nn.Module):
    def __init__(self, C, w):
        super().__init__()
        self.style = nn.Linear(w, 2 * C)
        with torch.no_grad():
            self.style.bias[:C] = 1
            self.style.bias[C:] = 0

    def forward(self, x, w):
        ys, yb = self.style(w).chunk(2, 1)
        m = x.mean((2, 3), keepdim=True)
        s = x.std((2, 3), keepdim=True, unbiased=False) + 1e-8
        return ys[:, :, None, None] * (x - m) / s + yb[:, :, None, None]


class NoiseInjection(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(1, C, 1, 1))

    def forward(self, x, noise=None):
        if noise is None:
            noise = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3])
        return x + self.scale * noise


def modulated_conv(x, weight, style, demod=True):
    B, Cin, H, W = x.shape
    Cout, _, k, _ = weight.shape
    w = weight[None] * style[:, None, :, None, None]
    if demod:
        w = w / torch.sqrt((w ** 2).sum((2, 3, 4), keepdim=True) + 1e-8)
    out = F.conv2d(x.reshape(1, B * Cin, H, W), w.reshape(B * Cout, Cin, k, k), padding=k // 2, groups=B)
    return out.view(B, Cout, H, W)


def truncate(w, w_avg, psi):
    return w_avg + psi * (w - w_avg)
