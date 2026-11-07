import math
import random
from labcheck import close, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("numpy/torch kullanılmıyor", lambda: no_imports(src, {"numpy", "torch", "sklearn"}))

    def es():
        Rm = need(ns, "e_step")([1, 2, 4], [0.5, 0.5], [1, 4], [1, 1])
        close([r[0] for r in Rm], [0.989013, 0.817574, 0.010987], atol=1e-5)
        close([sum(r) for r in Rm], [1, 1, 1], what="satır toplamları")
        Rf = need(ns, "e_step")([500.0], [0.5, 0.5], [0, 1], [1, 1])
        ensure(all(math.isfinite(v) for v in Rf[0]), "uzak noktada NaN; log uzayında çalış")
        close(Rf[0], [0, 1], atol=1e-9, what="x = 500")
    R.test("e_step: elle örnek ve uzak noktada kararlı", es)

    def ms():
        Rm = need(ns, "e_step")([1, 2, 4], [0.5, 0.5], [1, 4], [1, 1])
        pis, mus, vs = need(ns, "m_step")([1, 2, 4], Rm)
        close([pis[0], mus[0]], [0.605858, 1.467978], atol=2e-5)
    R.test("m_step: π₁ = 0.606, μ₁ = 1.468", ms)

    def mono():
        rng = random.Random(0)
        xs = [rng.gauss(-2, 0.7) for _ in range(60)] + [rng.gauss(3, 1.2) for _ in range(90)]
        pis, mus, vs, hist = need(ns, "em")(xs, [0.5, 0.5], [0.0, 1.0], [1.0, 1.0], 30)
        ensure(len(hist) == 31, f"ll_geçmişi uzunluğu {len(hist)}, 31 bekleniyordu")
        ensure(all(b >= a - 1e-9 for a, b in zip(hist, hist[1:])), "log-likelihood bir iterasyonda düştü")
        o = sorted(range(2), key=lambda k: mus[k])
        close([mus[o[0]], mus[o[1]]], [-2, 3], atol=0.35, rtol=0, what="bulunan ortalamalar")
        return f"ll: {hist[0]:.1f} → {hist[-1]:.1f}; μ = {mus[o[0]]:.2f}, {mus[o[1]]:.2f}"
    R.test("em: log-likelihood monoton, ortalamaları buluyor", mono)
