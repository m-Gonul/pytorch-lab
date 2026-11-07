import torch


def equalize_batch(imgs):
    """imgs: (B, H, W) uint8. OpenCV varyantı. Döngü yok."""
    B = imgs.shape[0]
    flat = imgs.reshape(B, -1).long()
    N = flat.shape[1]
    idx = flat + 256 * torch.arange(B)[:, None]
    h = torch.bincount(idx.reshape(-1), minlength=256 * B).reshape(B, 256)
    c = h.cumsum(1)
    first = (h > 0).float().argmax(1, keepdim=True)
    cmin = h.gather(1, first)
    den = (N - cmin).clamp_min(1)
    lut = torch.round(255 * (c - cmin).double() / den.double()).clamp(0, 255)
    lut = torch.where(N == cmin, torch.arange(256, dtype=lut.dtype).expand(B, 256), lut).to(torch.uint8)
    return lut.gather(1, flat).reshape(imgs.shape)


imgs = torch.randint(30, 120, (4, 64, 64), dtype=torch.uint8)
print(equalize_batch(imgs).float().mean((1, 2)))
