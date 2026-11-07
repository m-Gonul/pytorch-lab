import numpy as np
from labcheck import close, max_loops, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("activation_stds: yalnız katman döngüsü", lambda: max_loops(src, 1, func="activation_stds"))
    R.test("torch kullanılmıyor", lambda: no_imports(src, {"torch"}))
    R.test("xavier_std, kaiming_std", lambda: close([need(ns, "xavier_std")(100, 100), need(ns, "kaiming_std")(100), need(ns, "xavier_std")(300, 100)], [0.1, 0.02 ** 0.5, 0.005 ** 0.5]))

    def ref(X, depth, width, std, act, seed):
        rng = np.random.default_rng(seed); H, o = X, []
        for _ in range(depth):
            H = act(H @ (rng.standard_normal((width, width)) * std)); o.append(H.std())
        return o
    relu = lambda z: np.maximum(0, z)
    X = np.random.default_rng(0).standard_normal((300, 128))
    R.test("activation_stds referansla aynı", lambda: close(need(ns, "activation_stds")(X, 5, 128, 0.1, np.tanh, np.random.default_rng(2)), ref(X, 5, 128, 0.1, np.tanh, 2)))

    def scen():
        f = need(ns, "activation_stds")
        rows = []
        cases = [("std = 1, ReLU", 1.0, relu), ("Xavier, ReLU", need(ns, "xavier_std")(128, 128), relu),
                 ("Kaiming, ReLU", need(ns, "kaiming_std")(128), relu), ("Xavier, tanh", need(ns, "xavier_std")(128, 128), np.tanh)]
        res = {}
        for name, s, a in cases:
            with np.errstate(over="ignore", invalid="ignore"):
                o = f(X, 30, 128, s, a, np.random.default_rng(3))
            res[name] = o
            rows.append([name, f"{o[0]:.3g}", f"{o[9]:.3g}", f"{o[-1]:.3g}"])
        R.table("Aktivasyon std'si, 30 katman (genişlik 128)", ["başlatma", "katman 1", "katman 10", "katman 30"], rows)
        k = res["Kaiming, ReLU"]
        ensure(0.3 < k[-1] / k[0] < 3, f"Kaiming + ReLU kararlı olmalı: {k[0]:.3g} → {k[-1]:.3g}")
        ensure(res["Xavier, ReLU"][-1] < 1e-3 * res["Xavier, ReLU"][0], "Xavier + ReLU sönmeli")
        ensure(not np.isfinite(res["std = 1, ReLU"][-1]) or res["std = 1, ReLU"][-1] > 1e10, "std = 1 patlamalı")
    R.test("Kaiming + ReLU kararlı, Xavier + ReLU sönüyor, std = 1 patlıyor", scen)
