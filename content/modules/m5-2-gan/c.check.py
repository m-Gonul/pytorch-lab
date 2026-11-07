import numpy as np
from labcheck import close, no_loops, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("döngü yok", lambda: no_loops(src))
    R.test("torch kullanılmıyor", lambda: no_imports(src, {"torch"}))
    rng = np.random.default_rng(0)
    p = rng.random(30); p[[3, 7]] = 0; p /= p.sum()
    q = rng.random(30); q[[7, 12]] = 0; q /= q.sum()

    def js():
        m = 0.5 * (p + q)
        ref = 0.5 * sum(a * np.log(a / b) for a, b in zip(p, m) if a > 0) + 0.5 * sum(a * np.log(a / b) for a, b in zip(q, m) if a > 0)
        close(need(ns, "jsd")(p, q), ref)
        close(ns["jsd"](p, p), 0.0, what="JSD(p, p)")
        close(ns["jsd"](np.array([1.0, 0]), np.array([0, 1.0])), np.log(2), what="ayrık destekler → log 2")
    R.test("jsd (sıfır kutular dahil)", js)

    def eqn():
        D = need(ns, "optimal_d")(p, q)
        ensure(np.all(np.isfinite(D)) and D[7] == 0.5, "p = q = 0 kutusunda 0.5")
        Dc = np.clip(D, 1e-12, 1 - 1e-12)
        close(need(ns, "value_expect")(p, q, Dc), -np.log(4) + 2 * ns["jsd"](p, q), atol=1e-8)
    R.test("V(D*) = −log 4 + 2·JSD", eqn)

    def maxim():
        D = np.clip(need(ns, "optimal_d")(p, q), 1e-9, 1 - 1e-9)
        v = need(ns, "value_expect")(p, q, D)
        for _ in range(20):
            Dp = np.clip(D + rng.normal(0, 0.05, D.shape), 1e-6, 1 - 1e-6)
            ensure(ns["value_expect"](p, q, Dp) <= v + 1e-12, "bozulmuş D daha yüksek değer verdi")
    R.test("D* gerçekten maksimum", maxim)

    def grads():
        sat, ns_ = need(ns, "gen_logit_grads")(np.array([-5.0, 0.0, 5.0]))
        close(sat, [-0.006693, -0.5, -0.993307], atol=1e-6, what="doymuş")
        close(ns_, [-0.993307, -0.5, -0.006693], atol=1e-6, what="doymamış")
        return f"logit −5'te (D emin: sahte) doymuş {sat[0]:.4f}, doymamış {ns_[0]:.4f}: ~150 kat"
    R.test("doymuş vs doymamış gradyan", grads)
