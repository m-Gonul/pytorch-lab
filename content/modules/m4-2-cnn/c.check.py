import numpy as np
from labcheck import close, max_loops, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("yalnız katman döngüleri (iç içe değil)", lambda: max_loops(src, 1, func="grad_norms"))
    R.test("torch kullanılmıyor", lambda: no_imports(src, {"torch"}))
    rng = np.random.default_rng(0)
    D = 12
    X = rng.standard_normal((5, D))
    Ws = [rng.standard_normal((D, D)) * 0.6 / np.sqrt(D) for _ in range(4)]

    def num(residual):
        def f(X0):
            h = X0
            for W in Ws:
                t = np.tanh(h @ W); h = h + t if residual else t
            return h.sum()
        g = np.zeros_like(X); hh = 1e-6
        for i in np.ndindex(X.shape):
            Xp, Xm = X.copy(), X.copy(); Xp[i] += hh; Xm[i] -= hh
            g[i] = (f(Xp) - f(Xm)) / (2 * hh)
        return np.linalg.norm(g)

    def gr():
        for res in (False, True):
            n = need(ns, "grad_norms")(X, Ws, res)
            ensure(len(n) == 5, f"L + 1 = 5 değer bekleniyordu, {len(n)} geldi")
            close(n[0], num(res), rtol=1e-5, what=f"‖g₀‖ (residual={res}) sayısal gradyanla")
            close(n[-1], np.sqrt(X.size), what="‖g_L‖ = ‖1‖")
    R.test("‖g₀‖ sayısal gradyanla aynı (düz ve residual)", gr)

    def comp():
        D2, L = 64, 20
        X2 = rng.standard_normal((32, D2))
        W2 = [rng.standard_normal((D2, D2)) * 0.5 / np.sqrt(D2) for _ in range(L)]
        p = need(ns, "grad_norms")(X2, W2, False); r = need(ns, "grad_norms")(X2, W2, True)
        ensure(p[0] / p[-1] < 1e-3, f"düz ağda gradyan sönmeli: oran {p[0] / p[-1]:.3g}")
        ensure(r[0] / r[-1] > 0.5, f"residual ağda gradyan korunmalı: oran {r[0] / r[-1]:.3g}")
        R.table("20 katman, ‖g_l‖ / ‖g_L‖", ["katman", "düz", "residual"],
                [[str(l), f"{p[l] / p[-1]:.2e}", f"{r[l] / r[-1]:.3f}"] for l in (0, 5, 10, 15, 20)])
    R.test("düz ağda sönme, residual'da koruma", comp)
