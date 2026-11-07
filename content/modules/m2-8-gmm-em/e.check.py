import math
import torch
from labcheck import close, max_loops, need, ensure, bench, fmt_time


def one(X, mus, iters, reg=1e-6):
    n, d = X.shape; K = mus.shape[0]
    pis = torch.full((K,), 1.0 / K, dtype=X.dtype); vs = X.var(0, unbiased=False).repeat(K, 1)
    lp = lambda pis, mus, vs: pis.log() - 0.5 * (torch.log(2 * math.pi * vs)[None] + (X[:, None] - mus[None]) ** 2 / vs[None]).sum(-1)
    for _ in range(iters):
        L = lp(pis, mus, vs)
        Rr = (L - torch.logsumexp(L, 1, keepdim=True)).exp()
        Nk = Rr.sum(0); pis = Nk / n; mus = Rr.T @ X / Nk[:, None]
        vs = (Rr[..., None] * (X[:, None] - mus[None]) ** 2).sum(0) / Nk[:, None] + reg
    return torch.logsumexp(lp(pis, mus, vs), 1).sum(), mus


def check(ns, R):
    src = ns["__source__"]
    R.test("yalnız iterasyon döngüsü", lambda: max_loops(src, 1, func="em_restarts_diag", why="başlatmalar üzerinde döngü olmamalı"))
    g = torch.Generator().manual_seed(0)
    X = torch.cat([torch.randn(60, 2, generator=g, dtype=torch.float64) + torch.tensor(c, dtype=torch.float64) for c in ([0.0, 0], [5, 0], [0, 5])])
    mus0 = X[torch.randint(0, 180, (8, 3), generator=g)]

    def eq():
        lls, (p, m, v) = need(ns, "em_restarts_diag")(X, mus0, 25)
        ensure(tuple(lls.shape) == (8,), f"lls shape {tuple(lls.shape)}")
        ref = [one(X, mus0[r], 25) for r in range(8)]
        close(lls, torch.stack([a for a, _ in ref]), rtol=1e-8)
        close(m, ref[int(lls.argmax())][1], rtol=1e-8, what="en iyi başlatmanın ortalamaları")
        return f"ll aralığı {float(lls.min()):.1f} – {float(lls.max()):.1f}"
    R.test("8 başlatma: tek tek EM ile aynı", eq)
    if "em_restarts_diag" not in ns:
        return
    Xb = torch.randn(1000, 4, generator=g)
    mb = Xb[torch.randint(0, 1000, (32, 5), generator=g)]
    t_loop = bench(lambda: [one(Xb, mb[r], 20) for r in range(32)], repeat=3, number=1)
    t_vec = bench(lambda: ns["em_restarts_diag"](Xb, mb, 20), repeat=3, number=1)
    R.table("32 başlatma × 20 EM iterasyonu (n = 1000, K = 5)", ["yöntem", "süre", "döngüye göre"],
            [["for r: EM", fmt_time(t_loop), "1×"], ["senin em_restarts_diag", fmt_time(t_vec), f"{t_loop / t_vec:,.1f}× hızlı"]])
