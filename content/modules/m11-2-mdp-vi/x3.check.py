import numpy as np
from labcheck import need, ensure


def check(ns, R_):
    def t():
        r, g = need(ns, "find_l2_counterexample")(20000, 0)
        ensure(r > g, f"L2 oranı {r:.3f}, γ = {g:.3f}: γ'yı aşan bir örnek bulunmalı")
        rng = np.random.default_rng(1)
        worst = 0
        for _ in range(5000):
            P = rng.random((2, 3, 3)) ** 4; P /= P.sum(-1, keepdims=True)
            R = rng.normal(size=(3, 2)); gg = rng.uniform(0.5, 0.99)
            V, U = rng.normal(size=3) * 5, rng.normal(size=3) * 5
            worst = max(worst, np.abs(need(ns, "h_star")(P, R, gg, V) - ns["h_star"](P, R, gg, U)).max() / np.abs(V - U).max() / gg)
        ensure(worst <= 1 + 1e-12, "sonsuz normda oran γ'yı aşmamalı")
        return f"L2'de oran {r:.3f} > γ = {g:.3f} (büzülme değil); ∞-normda en büyük oran/γ = {worst:.4f} ≤ 1"
    R_.test("büzülme normun seçimine bağlı", t)
