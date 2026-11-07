import math
import torch
import torch.nn.functional as F


def warp(img, flow):
    H, W = img.shape
    ys, xs = torch.meshgrid(torch.arange(H, dtype=img.dtype), torch.arange(W, dtype=img.dtype), indexing="ij")
    g = torch.stack([(xs + flow[0]) / (W - 1) * 2 - 1, (ys + flow[1]) / (H - 1) * 2 - 1], -1)[None]
    return F.grid_sample(img[None, None], g, align_corners=True, padding_mode="border")[0, 0]


def _box(x, k):
    return F.avg_pool2d(F.pad(x[None, None], (k // 2,) * 4, mode="replicate"), k, 1)[0, 0]


def _grad(x):
    gx = F.pad(x[None, None], (1, 1, 0, 0), mode="replicate")
    gy = F.pad(x[None, None], (0, 0, 1, 1), mode="replicate")
    return (gx[..., 2:] - gx[..., :-2])[0, 0] / 2, (gy[..., 2:, :] - gy[..., :-2, :])[0, 0] / 2


def lk_level(I0, I1, flow, win=7, iters=5):
    gx, gy = _grad(I0)
    a, b, c = _box(gx * gx, win), _box(gx * gy, win), _box(gy * gy, win)
    det = a * c - b * b
    lmin = (a + c) / 2 - torch.sqrt(((a - c) / 2) ** 2 + b * b)
    ok = lmin > 1e-2
    d = torch.where(ok, det, torch.ones_like(det))
    wgt = lmin.clamp_min(0)
    wb = _box(wgt, win) + 1e-9
    for _ in range(iters):
        It = warp(I1, flow) - I0
        p, q = _box(gx * It, win), _box(gy * It, win)
        du = torch.where(ok, (-c * p + b * q) / d, torch.zeros_like(d)).clamp(-1, 1)
        dv = torch.where(ok, (b * p - a * q) / d, torch.zeros_like(d)).clamp(-1, 1)
        flow = flow + torch.stack([du, dv])
        flow = torch.stack([_box(wgt * flow[0], win) / wb, _box(wgt * flow[1], win) / wb])
    return flow


def pyr_lk(I0, I1, levels=4, win=7, iters=5):
    P0, P1 = [I0], [I1]
    for _ in range(levels - 1):
        P0.append(F.avg_pool2d(P0[-1][None, None], 2)[0, 0])
        P1.append(F.avg_pool2d(P1[-1][None, None], 2)[0, 0])
    flow = torch.zeros(2, *P0[-1].shape, dtype=I0.dtype)
    for l in range(levels - 1, -1, -1):
        if flow.shape[1:] != P0[l].shape:
            flow = 2 * F.interpolate(flow[None], size=tuple(P0[l].shape), mode="bilinear", align_corners=False)[0]
        flow = lk_level(P0[l], P1[l], flow, win, iters)
    return flow


def fb_error(f01, f10):
    back = torch.stack([warp(f10[0], f01), warp(f10[1], f01)])
    return (f01 + back).norm(dim=0)


def fruc_mc(I0, I2, flow02):
    return 0.5 * (warp(I0, -0.5 * flow02) + warp(I2, 0.5 * flow02))


x = torch.rand(32, 32, dtype=torch.float64)
print(warp(x, torch.zeros(2, 32, 32, dtype=torch.float64)).allclose(x))
