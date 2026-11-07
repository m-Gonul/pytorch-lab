import math
import numpy as np
from labcheck import close, max_loops, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("döngü kuralları", lambda: (max_loops(src, 1, func="tree_reduce"), max_loops(src, 1, func="argmax_tree"), max_loops(src, 1, func="chain_product_tree")))
    rng = np.random.default_rng(0)

    def t():
        close(need(ns, "tree_reduce")(np.arange(1, 1001), np.add, 0), 500500, what="toplam")
        x = rng.normal(size=777)
        close(ns["tree_reduce"](x, np.maximum, -np.inf), x.max(), what="max")
        close(ns["tree_reduce"](x[:5], np.minimum, np.inf), x[:5].min(), what="tek uzunluk")
        ensure(need(ns, "argmax_tree")(x) == int(np.argmax(x)), "argmax")
        ensure(ns["argmax_tree"](np.array([1, 5, 5, 2, 5])) == 1, "eşitlikte ilk indeks")
        M = rng.normal(size=(37, 2, 2))
        ref = np.eye(2)
        for m in M:
            ref = ref @ m
        close(need(ns, "chain_product_tree")(M), ref, rtol=1e-9, what="sıralı matris çarpımı = ağaç")
        rows = []
        for n in (10 ** 4, 10 ** 6, 10 ** 7):
            v = rng.random(n).astype(np.float32)
            ref = math.fsum(v.astype(np.float64))
            a, b = need(ns, "naive_sum32")(v), need(ns, "tree_sum32")(v)
            rows.append([n, f"{abs(a - ref) / ref:.1e}", f"{abs(b - ref) / ref:.1e}"])
        ensure(float(rows[-1][2]) < float(rows[-1][1]) / 10, "ağaç toplamı çok daha doğru")
        R_.table("float32 toplamın göreli hatası (U[0,1) sayılar)", ["n", "sıralı", "ağaç (ikili)"], rows,
                 note="Sıralıda biriktirici büyüdükçe her yeni küçük sayı daha çok yuvarlanır (hata ~nε). Ağaçta her toplama benzer büyüklükleri toplar (~ε log n). GPU indirgemesi böylece 'yan etki olarak' daha doğrudur.")
    R_.test("ağaç indirgemeleri", t)
