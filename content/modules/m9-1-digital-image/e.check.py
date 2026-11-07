import math
import torch
import torch.nn.functional as F
from labcheck import close, max_loops, need, ensure, bench, fmt_time


def ref_one(img, A, t):
    H, W = img.shape
    ys, xs = torch.meshgrid(torch.arange(H, dtype=img.dtype), torch.arange(W, dtype=img.dtype), indexing="ij")
    S = torch.stack([xs, ys], -1) @ A.T + t
    grid = torch.stack([(2 * S[..., 0] + 1) / W - 1, (2 * S[..., 1] + 1) / H - 1], -1)[None]
    return F.grid_sample(img[None, None], grid, align_corners=False)[0, 0]


def check(ns, R):
    src = ns["__source__"]
    R.test("döngü yok", lambda: max_loops(src, 0, func="warp_batch"))
    g = torch.Generator().manual_seed(0)
    B = 16
    imgs = torch.rand(B, 40, 56, generator=g, dtype=torch.float64)
    th = torch.rand(B, generator=g, dtype=torch.float64) * 2 * math.pi
    sc = 0.7 + 0.6 * torch.rand(B, generator=g, dtype=torch.float64)
    A = torch.stack([torch.stack([th.cos(), -th.sin()], -1), torch.stack([th.sin(), th.cos()], -1)], 1) * sc[:, None, None]
    t = torch.randn(B, 2, generator=g, dtype=torch.float64) * 5

    def eq():
        out = need(ns, "warp_batch")(imgs, A, t)
        for i in range(B):
            close(out[i], ref_one(imgs[i], A[i], t[i]), atol=1e-9, what=f"görüntü {i}")
    R.test("her görüntü kendi dönüşümüyle", eq)
    if "warp_batch" not in ns:
        return
    Bb = 256
    big = torch.rand(Bb, 64, 64, generator=g)
    Ab, tb = A[:1].float().repeat(Bb, 1, 1), t[:1].float().repeat(Bb, 1)
    t_loop = bench(lambda: [ref_one(big[i], Ab[i], tb[i]) for i in range(Bb)], repeat=3, number=1)
    t_vec = bench(lambda: ns["warp_batch"](big, Ab, tb), repeat=3)
    R.table(f"{Bb} görüntü, 64 × 64", ["yöntem", "süre"], [["görüntü başına grid_sample", fmt_time(t_loop)], ["tek çağrı", fmt_time(t_vec)]],
            note="Veri artırmada (rastgele döndürme/ölçek) her örneğe ayrı dönüşüm böyle uygulanır.")
