import math
import torch
import torch.nn.functional as F


def _gk(s):
    r = int(math.ceil(4 * s))
    t = torch.arange(-r, r + 1, dtype=torch.float64)
    g = torch.exp(-t ** 2 / (2 * s * s))
    return g / g.sum()


def _blur(x, s):
    g = _gk(s)
    r = len(g) // 2
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), g.view(1, 1, 1, -1))
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), g.view(1, 1, -1, 1))


def dog_pyramid(img, octaves=4, s=3, sigma0=1.6):
    k = 2 ** (1 / s)
    base = _blur(img[None, None], sigma0)
    pyr = []
    for _ in range(octaves):
        L = [base]
        for i in range(1, s + 3):
            L.append(_blur(L[-1], math.sqrt((sigma0 * k ** i) ** 2 - (sigma0 * k ** (i - 1)) ** 2)))
        pyr.append(torch.cat([L[i + 1] - L[i] for i in range(s + 2)], 1)[0])
        base = L[s][..., ::2, ::2]
    return pyr


def detect(img, octaves=4, s=3, sigma0=1.6, thr=0.03):
    out = []
    for o, D in enumerate(dog_pyramid(img, octaves, s, sigma0)):
        D5 = D[None, None]
        mx = F.max_pool3d(D5, 3, 1, 1)
        mn = -F.max_pool3d(-D5, 3, 1, 1)
        m = ((D5 == mx) | (D5 == mn)) & (D5.abs() > thr)
        m[:, :, [0, -1]] = False
        m[..., [0, -1], :] = False
        m[..., :, [0, -1]] = False
        for i, y, x in m[0, 0].nonzero().tolist():
            out.append((x * 2 ** o, y * 2 ** o, sigma0 * 2 ** (o + i / s), D[i, y, x].item()))
    return out


x = torch.zeros(64, 64, dtype=torch.float64); x[28:36, 28:36] = 1
print(len(detect(x, octaves=2)))
