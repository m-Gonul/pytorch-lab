import torch
import torch.nn.functional as F


def running_max(x, k):
    L = x.shape[-1]
    r = k // 2
    xp = F.pad(x, (r, r + (-(L + 2 * r)) % k), value=-float("inf"))
    blocks = xp.reshape(*x.shape[:-1], -1, k)
    g = blocks.cummax(-1).values.reshape(*x.shape[:-1], -1)
    h = blocks.flip(-1).cummax(-1).values.flip(-1).reshape(*x.shape[:-1], -1)
    return torch.maximum(h[..., :L], g[..., k - 1:k - 1 + L])


def rect_dilate(img, kh, kw):
    t = running_max(img.float(), kw)
    t = running_max(t.transpose(-1, -2), kh).transpose(-1, -2)
    return t.to(torch.uint8)


def rect_erode(img, kh, kw):
    t = -running_max(-img.float(), kw)
    t = -running_max(-t.transpose(-1, -2), kh).transpose(-1, -2)
    return t.to(torch.uint8)


print(running_max(torch.tensor([1.0, 5, 2, 0, 3, 9, 1]), 3))
