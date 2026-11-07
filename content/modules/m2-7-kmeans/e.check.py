import torch
from labcheck import close, max_loops, need, ensure, bench, fmt_time


def one(X, C, iters):
    C = C.clone(); k = C.shape[0]
    for _ in range(iters):
        lab = torch.cdist(X, C).argmin(1)
        s = torch.zeros_like(C).index_add_(0, lab, X)
        c = torch.bincount(lab, minlength=k).to(X.dtype)
        C = torch.where(c[:, None] > 0, s / c.clamp_min(1)[:, None], C)
    return C, (torch.cdist(X, C) ** 2).min(1).values.sum()


def check(ns, R):
    src = ns["__source__"]
    R.test("yalnız iterasyon döngüsü", lambda: max_loops(src, 1, func="kmeans_restarts", why="başlatmalar üzerinde döngü olmamalı"))
    g = torch.Generator().manual_seed(0)
    X = torch.cat([torch.randn(40, 2, generator=g, dtype=torch.float64) + torch.tensor(c, dtype=torch.float64) for c in ([0.0, 0], [6, 0], [0, 6], [6, 6])])
    C0s = X[torch.randint(0, 160, (10, 4), generator=g)]

    def eq():
        best, J = need(ns, "kmeans_restarts")(X, C0s, 15)
        ensure(tuple(J.shape) == (10,), f"inertias shape {tuple(J.shape)}")
        ref = [one(X, C0s[r], 15) for r in range(10)]
        close(J, torch.stack([j for _, j in ref]), rtol=1e-9)
        close(best, ref[int(torch.stack([j for _, j in ref]).argmin())][0])
        return f"inertia aralığı {float(J.min()):.1f} – {float(J.max()):.1f}: başlatma önemli"
    R.test("10 başlatma: tek tek koşturmayla aynı", eq)
    if "kmeans_restarts" not in ns:
        return
    Xb = torch.randn(2000, 8, generator=g)
    C0b = Xb[torch.randint(0, 2000, (64, 10), generator=g)]
    t_loop = bench(lambda: [one(Xb, C0b[r], 10) for r in range(64)], repeat=3, number=1)
    t_vec = bench(lambda: ns["kmeans_restarts"](Xb, C0b, 10), repeat=3, number=1)
    R.table("64 başlatma × 10 iterasyon (n = 2000, k = 10)", ["yöntem", "süre", "döngüye göre"],
            [["for r: kmeans", fmt_time(t_loop), "1×"], ["senin kmeans_restarts", fmt_time(t_vec), f"{t_loop / t_vec:,.1f}× hızlı"]])
