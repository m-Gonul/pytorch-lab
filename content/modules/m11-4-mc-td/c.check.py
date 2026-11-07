import random
import numpy as np
from labcheck import close, max_loops, need, ensure


def walk(rng):
    s, out = 3, []
    while True:
        s2 = s + (1 if rng.random() < 0.5 else -1)
        out.append((s - 1, 1.0 if s2 == 6 else 0.0))
        if s2 in (0, 6):
            return out
        s = s2


def check(ns, R_):
    src = ns["__source__"]
    R_.test("yalnız tekrar döngüsü", lambda: (max_loops(src, 0, func="batch_mc"), max_loops(src, 1, func="batch_td")))
    AB = [[(0, 0.0), (1, 0.0)]] + [[(1, 1.0)]] * 6 + [[(1, 0.0)]]

    def ab():
        close(need(ns, "batch_mc")(AB, 2), np.array([0.0, 0.75]), what="toplu MC")
        close(need(ns, "batch_td")(AB, 2), np.array([0.75, 0.75]), atol=1e-6, what="toplu TD")
        close(need(ns, "certainty_equivalence")(AB, 2), np.array([0.75, 0.75]), atol=1e-12, what="kesinlik eşdeğerliği")
    R_.test("Sutton Örnek 6.4", ab)

    def rw():
        true = np.arange(1, 6) / 6
        rows = []
        for k in (5, 20, 80):
            mc_e = td_e = 0
            for seed in range(30):
                rng = random.Random(seed)
                eps = [walk(rng) for _ in range(k)]
                vt = ns["batch_td"](eps, 5, 0.002, 1e-9)
                close(vt, ns["certainty_equivalence"](eps, 5), atol=1e-5, what="toplu TD = kesinlik eşdeğerliği")
                mc_e += np.sqrt(((ns["batch_mc"](eps, 5) - true) ** 2).mean()); td_e += np.sqrt(((vt - true) ** 2).mean())
            rows.append([k, f"{mc_e / 30:.3f}", f"{td_e / 30:.3f}"])
        ensure(float(rows[0][2]) < float(rows[0][1]), "az veride toplu TD daha doğru olmalı")
        R_.table("Toplu güncelleme, rastgele yürüyüş (30 tekrar)", ["bölüm", "toplu MC RMS", "toplu TD RMS"], rows)
    R_.test("rastgele yürüyüşte toplu MC vs TD", rw)
