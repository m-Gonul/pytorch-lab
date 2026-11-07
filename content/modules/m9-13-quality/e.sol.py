import torch
import torch.nn.functional as F


def _g():
    t = torch.arange(-5, 6, dtype=torch.float64)
    g = torch.exp(-t ** 2 / (2 * 1.5 ** 2))
    return g / g.sum()


def _sym(a, r):
    return torch.cat([a[..., :r].flip(-1), a, a[..., -r:].flip(-1)], -1)


def _gauss(a):
    g = _g().to(a.dtype)
    a = F.conv2d(_sym(a, 5), g.view(1, 1, 1, -1))
    a = _sym(a.transpose(-1, -2), 5).transpose(-1, -2)
    return F.conv2d(a, g.view(1, 1, -1, 1))


def ms_ssim(x, y, R=255.0, weights=(0.0448, 0.2856, 0.3001, 0.2363, 0.1333)):
    X, Y = x[:, None], y[:, None]
    C1, C2 = (0.01 * R) ** 2, (0.03 * R) ** 2
    out = torch.ones(x.shape[0], dtype=x.dtype)
    M = len(weights)
    for j, w in enumerate(weights):
        ux, uy = _gauss(X), _gauss(Y)
        vx, vy, vxy = _gauss(X * X) - ux * ux, _gauss(Y * Y) - uy * uy, _gauss(X * Y) - ux * uy
        cs = (2 * vxy + C2) / (vx + vy + C2)
        if j < M - 1:
            out = out * torch.relu(cs[..., 5:-5, 5:-5].mean((1, 2, 3))) ** w
            X, Y = F.avg_pool2d(X, 2), F.avg_pool2d(Y, 2)
        else:
            l = (2 * ux * uy + C1) / (ux * ux + uy * uy + C1)
            out = out * torch.relu((l * cs)[..., 5:-5, 5:-5].mean((1, 2, 3))) ** w
    return out


x = torch.rand(2, 192, 192, dtype=torch.float64) * 255
print(ms_ssim(x, x))
