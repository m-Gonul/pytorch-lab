import torch
import torch.nn.functional as F


def adaptive_median_vec(img, smax=7):
    """img (H, W) uint8 tensör. Piksel döngüsü yok."""
    H, W = img.shape
    R = smax // 2
    P = F.pad(img[None, None].float(), (R, R, R, R), mode="replicate")
    z = img.float()
    out = torch.full_like(z, float("nan"))
    last_med = None
    for k in range(3, smax + 1, 2):
        off = R - k // 2
        win = F.unfold(P[:, :, off:off + H + k - 1, off:off + W + k - 1], k)[0].T.reshape(H, W, k * k)
        zmin, zmax = win.amin(-1), win.amax(-1)
        zmed = win.sort(-1).values[..., (k * k) // 2]
        passA = (zmin < zmed) & (zmed < zmax)
        val = torch.where((zmin < z) & (z < zmax), z, zmed)
        out = torch.where(torch.isnan(out) & passA, val, out)
        last_med = zmed
    out = torch.where(torch.isnan(out), last_med, out)
    return out.to(torch.uint8)


img = torch.randint(0, 256, (16, 16), dtype=torch.uint8)
print(adaptive_median_vec(img).shape)
