import torch
from labcheck import close, no_loops, need, ensure, bench, fmt_time


def weighted_stump(X, y, w):
    n, d = X.shape
    best = None
    for j in range(d):
        v = torch.unique(X[:, j])
        for t in ((v[:-1] + v[1:]) / 2).tolist():
            m = X[:, j] <= t
            WL, WR = w[m].sum(), w[~m].sum()
            if WL == 0 or WR == 0:
                continue
            SL, SR = (w * y)[m].sum(), (w * y)[~m].sum()
            gn = float(SL ** 2 / WL + SR ** 2 / WR)
            if best is None or gn > best[0] + 1e-9:
                best = (gn, j, t, float(SL / WL), float(SR / WR))
    return best[1:]


def check(ns, R):
    src = ns["__source__"]
    R.test("döngü yok", lambda: no_loops(src))
    g = torch.Generator().manual_seed(0)
    X = torch.rand(40, 3, generator=g, dtype=torch.float64)
    y = torch.sin(6 * X[:, 0]) + X[:, 2] + 0.1 * torch.randn(40, generator=g, dtype=torch.float64)

    def eq():
        gen = torch.Generator().manual_seed(5)
        j, t, l, r = need(ns, "bagged_stumps")(X, y, 12, gen)
        ensure(tuple(j.shape) == (12,), f"j shape {tuple(j.shape)}")
        idx = torch.randint(40, (12, 40), generator=torch.Generator().manual_seed(5))
        for b in range(12):
            w = torch.bincount(idx[b], minlength=40).double()
            jj, tt, ll, rr = weighted_stump(X, y, w)
            ensure(int(j[b]) == jj, f"b={b}: özellik {int(j[b])}, {jj} bekleniyordu")
            close([float(t[b]), float(l[b]), float(r[b])], [tt, ll, rr], what=f"b={b}")
        return "12 bootstrap stump'ın her biri tek tek oturtulanla aynı (aynı generator dizisi)"
    R.test("her bootstrap stump referansla aynı", eq)

    def pred():
        m = need(ns, "bagged_stumps")(X, y, 200, torch.Generator().manual_seed(1))
        p = need(ns, "bagged_predict")(m, X)
        ensure(tuple(p.shape) == (40,), f"shape {tuple(p.shape)}")
        j, t, l, r = m
        manual = torch.stack([torch.where(X[:, int(j[b])] <= t[b], l[b], r[b]) for b in range(200)]).mean(0)
        close(p, manual)
    R.test("bagged_predict = stump tahminlerinin ortalaması", pred)
    if "bagged_stumps" not in ns:
        return
    Xb = torch.rand(300, 8, generator=g); yb = torch.sin(6 * Xb[:, 0]) + Xb[:, 1]

    def loop():
        idx = torch.randint(300, (50, 300), generator=torch.Generator().manual_seed(0))
        for b in range(50):
            w = torch.bincount(idx[b], minlength=300).float()
            Xs, order = torch.sort(Xb, 0); Ws = w[order]; Ss = (w * yb)[order]
            WL, SL = Ws.cumsum(0)[:-1], Ss.cumsum(0)[:-1]
            (SL ** 2 / WL.clamp_min(1e-9) + (Ss.sum(0) - SL) ** 2 / (Ws.sum(0) - WL).clamp_min(1e-9)).argmax()
    t_loop = bench(loop, repeat=3, number=1)
    t_vec = bench(lambda: ns["bagged_stumps"](Xb, yb, 50, torch.Generator().manual_seed(0)), repeat=3)
    R.table("50 bootstrap stump (n = 300, d = 8)", ["yöntem", "süre", "döngüye göre"],
            [["for b: vektörize stump (B sıralama)", fmt_time(t_loop), "1×"], ["senin bagged_stumps (1 sıralama)", fmt_time(t_vec), f"{t_loop / t_vec:,.1f}× hızlı"]])
