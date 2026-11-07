import numpy as np
from labcheck import need, ensure


def check(ns, R_):
    def t():
        rows = []
        best = {}
        for n in (1, 2, 4, 8, 16):
            row = [n]
            for a in (0.1, 0.2, 0.4, 0.8):
                e = np.mean([need(ns, "nstep_td")(need(ns, "episodes19")(10, s), n, a) for s in range(20)])
                row.append(f"{e:.3f}"); best[(n, a)] = e
            rows.append(row)
        n_best = min(best, key=best.get)
        ensure(1 < n_best[0] < 16, f"en iyi n arada olmalı, bulunan {n_best}")
        R_.table("19 durumlu yürüyüş, 10 bölüm, 20 tekrar ortalama RMS", ["n", "α = 0.1", "α = 0.2", "α = 0.4", "α = 0.8"], rows,
                 note=f"En iyi (n, α) = {n_best}: ne tek adım ne de MC — ara n en iyidir (Sutton Şekil 7.2).")
    R_.test("n ve α taraması", t)
