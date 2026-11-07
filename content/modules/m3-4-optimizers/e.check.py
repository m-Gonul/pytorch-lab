import math
import torch
from labcheck import close, max_loops, need, ensure, bench, fmt_time


@torch.no_grad()
def loop_step(params, grads, ms, vs, t, lr, b1, b2, eps, wd):
    for p, g, m, v in zip(params, grads, ms, vs):
        p.mul_(1 - lr * wd); m.mul_(b1).add_(g, alpha=1 - b1); v.mul_(b2).addcmul_(g, g, value=1 - b2)
        p.addcdiv_(m, v.sqrt() / math.sqrt(1 - b2 ** t) + eps, value=-lr / (1 - b1 ** t))


def check(ns, R):
    src = ns["__source__"]
    R.test("parametreler üzerinde döngü yok", lambda: max_loops(src, 0, func="adamw_foreach_"))
    g = torch.Generator().manual_seed(0)
    shapes = [(3, 4), (4,), (7, 2, 2), (1,)]

    def eq():
        P1 = [torch.randn(s, generator=g, dtype=torch.float64) for s in shapes]
        P2 = [p.clone() for p in P1]
        M1, V1 = [torch.zeros_like(p) for p in P1], [torch.zeros_like(p) for p in P1]
        M2, V2 = [torch.zeros_like(p) for p in P1], [torch.zeros_like(p) for p in P1]
        for t in range(1, 11):
            G = [torch.randn(s, generator=g, dtype=torch.float64) for s in shapes]
            need(ns, "adamw_foreach_")(P1, G, M1, V1, t, 0.01, 0.9, 0.999, 1e-8, 0.05)
            loop_step(P2, G, M2, V2, t, 0.01, 0.9, 0.999, 1e-8, 0.05)
        for a, b in zip(P1, P2):
            close(a, b, rtol=1e-12, atol=1e-14)
        for a, b in zip(V1, V2):
            close(a, b, rtol=1e-12, atol=1e-14, what="exp_avg_sq")
    R.test("10 adım: döngülü AdamW ile aynı (yerinde)", eq)
    if "adamw_foreach_" not in ns:
        return
    P = [torch.randn(64) for _ in range(500)]
    G = [torch.randn(64) for _ in range(500)]
    M, V = [torch.zeros(64) for _ in P], [torch.zeros(64) for _ in P]
    t_loop = bench(lambda: loop_step(P, G, M, V, 1, 1e-3, 0.9, 0.999, 1e-8, 0.01), repeat=5)
    t_fe = bench(lambda: ns["adamw_foreach_"](P, G, M, V, 1, 1e-3, 0.9, 0.999, 1e-8, 0.01), repeat=5)
    R.table("AdamW adımı, 500 tensör × 64 eleman", ["yöntem", "süre", "döngüye göre"],
            [["for p in params (≈ 3000 kernel çağrısı)", fmt_time(t_loop), "1×"], ["senin adamw_foreach_", fmt_time(t_fe), f"{t_loop / t_fe:,.1f}× hızlı"]],
            note="GPU'da fark çok daha büyük: foreach işlemleri tensörleri tek bir kernel'da toplar (multi-tensor apply).")
