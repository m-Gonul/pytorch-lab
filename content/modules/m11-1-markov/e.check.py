import math
import torch
from labcheck import close, no_loops, need, ensure, bench, fmt_time


def check(ns, R):
    src = ns["__source__"]
    R.test("döngü yok", lambda: no_loops(src))

    def var():
        W = need(ns, "wiener")(20000, 500, 2.0, 1)
        ensure(tuple(W.shape) == (20000, 501) and bool((W[:, 0] == 0).all()), "şekil, W0 = 0")
        for k in (100, 250, 500):
            t = 2.0 * k / 500
            close(W[:, k].var().item(), t, rtol=0.05, what=f"Var W({t})")
        close(torch.cov(torch.stack([W[:, 100], W[:, 400]])).numpy()[0, 1], 0.4, atol=0.03, what="Cov(W(0.4), W(1.6)) = min(s, t)")
    R.test("Wiener: Var W(t) = t, Cov = min(s, t)", var)

    def qv():
        rows = []
        for n in (10, 100, 1000):
            q = need(ns, "quadratic_variation")(ns["wiener"](5000, n, 1.0, 2))
            rows.append([n, f"{q.mean().item():.4f}", f"{q.std().item():.4f}"])
        ensure(float(rows[-1][2]) < float(rows[0][2]) / 5, "karesel değişimin varyansı adım sayısıyla sıfıra gitmeli")
        R.table("Karesel değişim Σ(ΔW)², t = 1", ["adım", "ortalama", "std"], rows, note="Pürüzsüz fonksiyonlarda karesel değişim 0'dır; Wiener sürecinde deterministik olarak t'dir (Itô hesabının temeli).")
    R.test("quadratic_variation", qv)

    def refl():
        W = ns["wiener"](20000, 2000, 1.0, 3)
        a, b = need(ns, "max_exceed")(W, 1.0)
        close(b, 2 * (1 - 0.5 * (1 + math.erf(1 / math.sqrt(2)))), atol=0.015, what="2P(W ≥ 1)")
        ensure(abs(a - b) < 0.03, f"yansıma ilkesi: P(max ≥ 1) = {a:.3f}, 2P(W ≥ 1) = {b:.3f}")
    R.test("yansıma ilkesi", refl)
    if "wiener" not in ns:
        return

    def loop():
        g = torch.Generator().manual_seed(0)
        out = torch.zeros(2000, 501, dtype=torch.float64)
        for t in range(500):
            out[:, t + 1] = out[:, t] + torch.randn(2000, generator=g, dtype=torch.float64) * (1 / 500) ** 0.5
        return out
    R.table("2000 yol × 500 adım", ["yöntem", "süre"], [["adım döngüsü", fmt_time(bench(loop, repeat=1, number=1))], ["cumsum", fmt_time(bench(lambda: ns["wiener"](2000, 500), repeat=3))]])
