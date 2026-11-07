import torch
import torch.nn.functional as F
from labcheck import close, no_loops, need, ensure, bench, fmt_time


def check(ns, R):
    src = ns["__source__"]
    R.test("döngü yok", lambda: no_loops(src))
    g = torch.Generator().manual_seed(0)
    x = torch.randn(3, 8, 5, 5, generator=g, dtype=torch.float64) * 2 + 1
    gm, bt = torch.rand(8, generator=g, dtype=torch.float64) + 0.5, torch.randn(8, generator=g, dtype=torch.float64)
    gn = lambda *a: need(ns, "group_norm")(*a)
    R.test("group_norm = F.group_norm (G = 4)", lambda: close(gn(x, 4, gm, bt), F.group_norm(x, 4, gm, bt)))
    R.test("G = C → instance_norm", lambda: close(gn(x, 8, torch.ones(8, dtype=x.dtype), torch.zeros(8, dtype=x.dtype)), F.instance_norm(x)))
    R.test("G = 1 → layer_norm (C, H, W)", lambda: close(gn(x, 1, torch.ones(8, dtype=x.dtype), torch.zeros(8, dtype=x.dtype)), F.layer_norm(x, x.shape[1:])))
    R.test("norm_over(x, (0, 2, 3)) → batch_norm (eğitim)", lambda: close(need(ns, "norm_over")(x, (0, 2, 3)), F.batch_norm(x, None, None, training=True)))
    if "group_norm" not in ns:
        return
    xb = torch.randn(32, 64, 16, 16, generator=g)
    o, z = torch.ones(64), torch.zeros(64)

    def loop():
        out = torch.empty_like(xb)
        for k in range(8):
            s = slice(8 * k, 8 * k + 8)
            blk = xb[:, s]
            mu = blk.mean((1, 2, 3), keepdim=True); var = blk.var((1, 2, 3), keepdim=True, unbiased=False)
            out[:, s] = (blk - mu) / torch.sqrt(var + 1e-5)
        return out
    t_loop = bench(loop, repeat=3)
    t_vec = bench(lambda: ns["group_norm"](xb, 8, o, z), repeat=3)
    t_ref = bench(lambda: F.group_norm(xb, 8, o, z), repeat=3)
    R.table("GroupNorm, (32, 64, 16, 16), G = 8", ["yöntem", "süre"],
            [["for grup: dilimle + normalize", fmt_time(t_loop)], ["senin group_norm (reshape)", fmt_time(t_vec)], ["F.group_norm (fused kernel)", fmt_time(t_ref)]],
            note="Fused kernel ara tensörleri (mean, var, x − μ) belleğe yazmadan tek geçişte hesaplar; M6.5/M7.2'de bunu kendin yapacaksın.")
