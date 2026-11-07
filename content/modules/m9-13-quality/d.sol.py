import math
import torch
import torch.nn.functional as F


def _g():
    t = torch.arange(-5, 6, dtype=torch.float64)
    g = torch.exp(-t ** 2 / (2 * 1.5 ** 2))
    return g / g.sum()


def _sym(a, r):
    return torch.cat([a[..., :r].flip(-1), a, a[..., -r:].flip(-1)], -1)


def gauss(a):
    g = _g().to(a.dtype)
    a = F.conv2d(_sym(a, 5), g.view(1, 1, 1, -1))
    a = _sym(a.transpose(-1, -2), 5).transpose(-1, -2)
    return F.conv2d(a, g.view(1, 1, -1, 1))


def ssim_t(x, y, R=255.0):
    X, Y = x[None, None], y[None, None]
    ux, uy = gauss(X), gauss(Y)
    vx, vy, vxy = gauss(X * X) - ux * ux, gauss(Y * Y) - uy * uy, gauss(X * Y) - ux * uy
    C1, C2 = (0.01 * R) ** 2, (0.03 * R) ** 2
    S = (2 * ux * uy + C1) * (2 * vxy + C2) / ((ux * ux + uy * uy + C1) * (vx + vy + C2))
    return S[0, 0, 5:-5, 5:-5].mean()


def mse_sphere(ref, mse, sign, steps=200, lr=2.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    e = torch.randn(ref.shape, generator=g, dtype=ref.dtype)
    e = e / e.pow(2).mean().sqrt() * math.sqrt(mse)
    for _ in range(steps):
        e.requires_grad_(True)
        gr, = torch.autograd.grad(ssim_t(ref, ref + e), e)
        with torch.no_grad():
            gr = gr - (gr * e).sum() / (e * e).sum() * e
            e = e + sign * lr * gr / (gr.pow(2).mean().sqrt() + 1e-12)
            e = e / e.pow(2).mean().sqrt() * math.sqrt(mse)
    return (ref + e).detach()


x = torch.rand(64, 64, dtype=torch.float64) * 255
print(ssim_t(x, x))
