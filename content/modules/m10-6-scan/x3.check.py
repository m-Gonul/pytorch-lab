import numpy as np
from labcheck import close, max_loops, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("döngüsüz", lambda: max_loops(src, 0, func="counting_sort"))

    def t():
        rng = np.random.default_rng(0)
        k = rng.integers(0, 16, 5000)
        s, perm = need(ns, "counting_sort")(k, 16)
        close(s, np.sort(k), what="sıralı")
        close(perm, np.argsort(k, kind="stable"), what="kararlı permütasyon")
        R_.note("Histogram (M10.5) + dışlayıcı tarama (bu modül) + dağıtım: GPU taban sıralamasının her basamağı. Kararlılık, LSD taban sıralamasının çok basamakta doğru çalışmasının şartıdır.")
    R_.test("sayma sıralaması", t)
