import random
import torch
from labcheck import close, no_loops, need, ensure, bench, fmt_time


def py_best(X, y):
    n, d = len(X), len(X[0])
    def G(s):
        if not s:
            return 0.0
        c = {}
        for v in s:
            c[v] = c.get(v, 0) + 1
        return 1 - sum((k / len(s)) ** 2 for k in c.values())
    best = None
    for j in range(d):
        vals = sorted(set(r[j] for r in X))
        for a, b in zip(vals, vals[1:]):
            t = (a + b) / 2
            L = [y[i] for i in range(n) if X[i][j] <= t]; Rr = [y[i] for i in range(n) if X[i][j] > t]
            s = len(L) / n * G(L) + len(Rr) / n * G(Rr)
            if best is None or s < best[2]:
                best = (j, t, s)
    return best


def check(ns, R):
    src = ns["__source__"]
    R.test("döngü yok", lambda: no_loops(src))
    f = lambda *a: need(ns, "best_split_torch")(*a)
    g = torch.Generator().manual_seed(0)
    X = torch.randn(80, 4, generator=g, dtype=torch.float64).round(decimals=2)
    y = torch.randint(0, 3, (80,), generator=g)

    def unw():
        got, want = f(X, y, 3), py_best(X.tolist(), y.tolist())
        ensure(got[0] == want[0], f"özellik {got[0]}, {want[0]} bekleniyordu"); close(got[1:], want[1:])
    R.test("ağırlıksız: saf Python referansla aynı", unw)

    def wt():
        w = torch.randint(1, 4, (80,), generator=g)
        got = f(X, y, 3, w.double())
        Xd, yd = X.repeat_interleave(w, 0), y.repeat_interleave(w)
        want = py_best(Xd.tolist(), yd.tolist())
        ensure(got[0] == want[0], f"özellik {got[0]}, {want[0]} bekleniyordu"); close(got[1:], want[1:])
        return "ağırlık k = satırı k kez kopyalamak"
    R.test("tamsayı ağırlık = satır kopyalama", wt)

    def hand():
        r = f(torch.tensor([[1.0], [2], [3], [4], [5], [6]]), torch.tensor([0, 0, 1, 1, 1, 0]), 2, torch.tensor([1.0, 1, 1, 1, 1, 10]))
        close(r[1], 5.5, what="son örneğe ağırlık 10 verince eşik")
    R.test("ağır örnek bölünmeyi kendi yanına çekiyor", hand)
    if "best_split_torch" not in ns:
        return
    rng = random.Random(0)
    Xb = torch.randn(2000, 20, generator=g)
    yb = torch.randint(0, 2, (2000,), generator=g)
    Xl, yl = Xb[:300].tolist(), yb[:300].tolist()
    t_py = bench(lambda: py_best(Xl, yl), repeat=1, number=1)
    t_vec = bench(lambda: ns["best_split_torch"](Xb[:300], yb[:300], 2), repeat=3)
    t_big = bench(lambda: ns["best_split_torch"](Xb, yb, 2), repeat=3)
    R.table("En iyi bölünme araması (d = 20)", ["yöntem", "n", "süre"],
            [["saf Python (her eşikte yeniden say)", "300", fmt_time(t_py)],
             ["senin best_split_torch", "300", fmt_time(t_vec)],
             ["senin best_split_torch", "2000", fmt_time(t_big)]],
            note=f"n = 300'de {t_py / t_vec:,.0f}× hızlı. Saf Python O(n²·d); sıralama + cumsum O(n log n · d).")
