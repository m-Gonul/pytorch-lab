import torch
import torch.nn.functional as F


def canny_torch(imgs, lo, hi, l2=False):
    """imgs (B, H, W) uint8 → (B, H, W) uint8 (0/255)."""
    if lo > hi:
        lo, hi = hi, lo
    B, H, W = imgs.shape
    k = torch.tensor([[[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]], [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]])[:, None]
    d = F.conv2d(F.pad(imgs[:, None].float(), (1, 1, 1, 1), mode="replicate"), k).round().long()
    dx, dy = d[:, 0], d[:, 1]
    if l2:
        m = dx * dx + dy * dy
        lo, hi = int(lo * lo // 1), int(hi * hi // 1)
    else:
        m = dx.abs() + dy.abs()
        lo, hi = int(lo // 1), int(hi // 1)
    P = F.pad(m, (1, 1, 1, 1))
    sh = lambda a, b: P[:, 1 + a:1 + a + H, 1 + b:1 + b + W]
    ax, ay = dx.abs(), dy.abs() << 15
    tg22x = ax * 13573
    tg67x = tg22x + (ax << 16)
    horiz = ay < tg22x
    vert = ~horiz & (ay > tg67x)
    diag = ~(horiz | vert)
    pos = (dx ^ dy) >= 0
    d1 = torch.where(pos, sh(-1, -1), sh(-1, 1))
    d2 = torch.where(pos, sh(1, 1), sh(1, -1))
    keep = (horiz & (m > sh(0, -1)) & (m >= sh(0, 1))) | (vert & (m > sh(-1, 0)) & (m >= sh(1, 0))) | (diag & (m > d1) & (m > d2))
    cand = keep & (m > lo)
    edges = cand & (m > hi)
    while True:
        grown = cand & (F.max_pool2d(edges[:, None].float(), 3, 1, 1)[:, 0] > 0)
        if torch.equal(grown, edges):
            break
        edges = grown
    return edges.to(torch.uint8) * 255


imgs = torch.randint(0, 256, (2, 32, 32), dtype=torch.uint8)
print(canny_torch(imgs, 50, 150).shape)
