import numpy as np
from labcheck import need, ensure


def check(ns, R_):
    def t():
        ensure(need(ns, "separable_error")(21, 3.0) < 1e-15, "Gauss ayrılabilir")
        x = np.arange(15) - 7
        X, Y = np.meshgrid(x, x)
        s = 2.5
        log = (X ** 2 + Y ** 2 - 2 * s ** 2) * np.exp(-(X ** 2 + Y ** 2) / (2 * s ** 2))
        disk = ((X ** 2 + Y ** 2) <= 36).astype(float)
        rnd = np.random.default_rng(0).normal(size=(15, 15))
        rows = []
        for name, k in [("Gauss σ = 2.5", np.exp(-(X ** 2 + Y ** 2) / (2 * s ** 2))), ("LoG", log), ("disk r = 6", disk), ("rastgele", rnd)]:
            errs = [need(ns, "svd_rank_approx")(k, r)[1] for r in (1, 2, 3, 5)]
            rows.append([name] + [f"{e:.1e}" for e in errs])
        ensure(float(rows[1][2]) < 1e-12, "LoG = iki ayrılabilir terimin toplamı (rank 2)")
        c = need(ns, "cost")(15, 2)
        ensure(c == (60, 225), "maliyet")
        R_.table("15×15 filtrelerde rank-r ayrılabilir yaklaşımın göreli hatası", ["filtre", "r = 1", "r = 2", "r = 3", "r = 5"], rows,
                 note="Maliyet r bileşen için 2Kr, tam 2B için K². LoG = (x² − 2σ²)g(x)g(y) + g(x)(y² − ...)g(y): tam olarak rank 2. Disk gibi dairesel simetrik ama Gauss olmayan çekirdekler birkaç bileşenle iyi yaklaşır; rastgele çekirdek tam ranklıdır.")
    R_.test("ayrılabilirlik", t)
