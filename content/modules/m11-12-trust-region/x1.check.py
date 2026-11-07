import math
from labcheck import need, ensure


def check(ns, R_):
    def t():
        nw = need(ns, "newton")([1.0, 0.01], 20)
        tr = need(ns, "trust_region_saddle")([1.0, 0.01], 60)
        fN, fT = ns["f"](nw[-1]), ns["f"](tr[-1])
        ensure(math.hypot(*nw[-1]) < 1e-6, "Newton eyer noktasına (0,0) gitmeli")
        ensure(abs(fT + 1) < 1e-6 and abs(abs(tr[-1][1]) - math.sqrt(2)) < 1e-4, "güven bölgesi minimuma (0, ±√2)")
        R_.table("f = x² − y² + y⁴/4, başlangıç (1, 0.01)", ["yöntem", "son nokta", "f"], [["Newton", f"({nw[-1][0]:.2e}, {nw[-1][1]:.2e})", f"{fN:.4f}"], ["güven bölgesi", f"({tr[-1][0]:.2e}, {tr[-1][1]:.4f})", f"{fT:.4f}"]],
                 note="Newton ikinci dereceden modelin durağan noktasına gider — H tanımsızsa bu eyer olabilir. Güven bölgesi modeli en aza indirir: negatif eğrilik yönünde bölge sınırına kadar gider ve eyerden kaçar.")
    R_.test("eyer noktası", t)
