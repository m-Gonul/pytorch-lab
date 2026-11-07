import numpy as np
from labcheck import close, max_loops, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("döngüsüz", lambda: (max_loops(src, 0, func="reduce_by_key"), max_loops(src, 0, func="segmented_max")))

    def t():
        rng = np.random.default_rng(0)
        k = rng.integers(0, 50, 10000)
        v = rng.normal(size=10000)
        uk, s = need(ns, "reduce_by_key")(k, v)
        close(uk, np.unique(k), what="anahtarlar")
        close(s, np.array([v[k == a].sum() for a in np.unique(k)]), what="toplamlar")
        flags = np.zeros(20, int)
        flags[[0, 3, 4, 11]] = 1
        x = rng.normal(size=20)
        close(need(ns, "segmented_max")(x, flags), np.array([x[0:3].max(), x[3:4].max(), x[4:11].max(), x[11:].max()]), what="segment max")
        R_.note("Anahtara göre indirgeme = sırala + segmentli indirgeme: veritabanlarında GROUP BY, grafiklerde düğüm başına toplama (scatter-add, GNN), seyrek matris-vektör çarpımında satır toplamları. GPU'da segment sınırları bir taramayla (M10.6) bulunur.")
    R_.test("segmentli indirgeme", t)
