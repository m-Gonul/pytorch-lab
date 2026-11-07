import itertools
import torch
import torch.nn.functional as F


def nlm(img, h, tw=7, sw=21, sigma=0.0):
    H, W = img.shape
    t, s = tw // 2, sw // 2
    P = F.pad(img[None, None], (t + s,) * 4, mode="reflect")[0, 0]
    base = P[s:s + H + 2 * t, s:s + W + 2 * t]
    num = torch.zeros_like(img)
    den = torch.zeros_like(img)
    for dy, dx in itertools.product(range(-s, s + 1), repeat=2):
        sh = P[s + dy:s + dy + H + 2 * t, s + dx:s + dx + W + 2 * t]
        d2 = F.avg_pool2d(((base - sh) ** 2)[None, None], tw, stride=1)[0, 0]
        w = torch.exp(-torch.clamp(d2 - 2 * sigma ** 2, min=0) / (h * h))
        num += w * sh[t:t + H, t:t + W]
        den += w
    return num / den


print(nlm(torch.rand(24, 24) * 255, 10.0, 3, 5).shape)
