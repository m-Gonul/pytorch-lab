import math
import numpy as np
import torch
import torch.nn.functional as F


def describe(img, kps):
    I = torch.as_tensor(np.asarray(img, np.float64))[None, None]
    H, W = I.shape[-2:]
    N = len(kps)
    x, y, s, th = kps.T
    t = torch.arange(16, dtype=torch.float64) - 7.5
    v, u = torch.meshgrid(t, t, indexing="ij")
    step = (3 * s / 4)[:, None, None]
    c, sn = torch.cos(th)[:, None, None], torch.sin(th)[:, None, None]

    def at(du, dv):
        uu, vv = u + du, v + dv
        px = x[:, None, None] + step * (c * uu - sn * vv)
        py = y[:, None, None] + step * (sn * uu + c * vv)
        g = torch.stack([px / (W - 1) * 2 - 1, py / (H - 1) * 2 - 1], -1)
        return F.grid_sample(I.expand(N, 1, H, W), g, align_corners=True, padding_mode="border")[:, 0]

    gu, gv = at(1, 0) - at(-1, 0), at(0, 1) - at(0, -1)
    mag = torch.hypot(gu, gv) * torch.exp(-(u ** 2 + v ** 2) / 128)
    ang = torch.atan2(gv, gu) % (2 * math.pi)
    ob = (ang / (2 * math.pi / 8)).floor().long().clamp(max=7)
    cell = (torch.div(v + 7.5, 4, rounding_mode="floor") * 4 + torch.div(u + 7.5, 4, rounding_mode="floor")).long()
    idx = (cell * 8)[None] + ob
    d = torch.zeros(N, 128, dtype=torch.float64).scatter_add_(1, idx.reshape(N, -1), mag.reshape(N, -1))
    d = d / d.norm(dim=1, keepdim=True).clamp_min(1e-12)
    d = d.clamp(max=0.2)
    return d / d.norm(dim=1, keepdim=True).clamp_min(1e-12)


from skimage import data
kps = torch.tensor([[256.0, 256.0, 2.0, 0.3]], dtype=torch.float64)
print(describe(data.camera(), kps).shape)
