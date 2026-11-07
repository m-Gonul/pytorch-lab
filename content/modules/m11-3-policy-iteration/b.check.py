import numpy as np
from labcheck import close, no_imports, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("numpy kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))

    def st():
        ensure(need(ns, "grid_step")(1, 0) == (1, -1.0) and ns["grid_step"](1, 1) == (5, -1.0) and ns["grid_step"](4, 2) == (4, -1.0), "duvarlar")
    R_.test("grid_step", st)

    def ev():
        V, sw = need(ns, "evaluate")([[0.25] * 4 for _ in range(16)])
        want = [0, -14, -20, -22, -14, -18, -20, -20, -20, -20, -18, -14, -22, -20, -14, 0]
        close(np.array(V), np.array(want, float), atol=1e-6, what="Sutton Şekil 4.1 (k = ∞)")
        return f"{sw} yerinde tarama"
    R_.test("rastgele politikanın değeri", ev)

    def pi():
        V, p, n = need(ns, "policy_iteration")()
        dist = [min(r + c, 6 - r - c) for r, c in (divmod(s, 4) for s in range(16))]
        close(np.array(V), -np.array(dist, float), atol=1e-6, what="en iyi değer = −(en yakın köşeye uzaklık)")
        ensure(n <= 4, f"{n} iyileştirme")
        arrows = "↑↓←→"
        grid = ["".join("■" if s in (0, 15) else arrows[p[s].index(1.0)] for s in range(r * 4, r * 4 + 4)) for r in range(4)]
        return f"{n} iyileştirme; politika: " + " / ".join(grid)
    R_.test("policy_iteration", pi)
