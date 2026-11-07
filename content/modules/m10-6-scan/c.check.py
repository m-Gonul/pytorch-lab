import numpy as np
from labcheck import close, max_loops, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("döngü kuralları", lambda: (max_loops(src, 1, func="ks_scan"), max_loops(src, 2, func="bk_scan"), max_loops(src, 0, func="compact"), max_loops(src, 1, func="radix_sort"), max_loops(src, 1, func="linear_recurrence")))
    rng = np.random.default_rng(0)

    def t():
        for n in (1, 8, 1000, 4096):
            x = rng.integers(-50, 50, n)
            close(need(ns, "ks_scan")(x), np.cumsum(x), what=f"KS n = {n}")
        for n in (2, 8, 1024):
            x = rng.integers(-50, 50, n)
            close(need(ns, "bk_scan")(x), np.cumsum(x), what=f"BK n = {n}")
        x = rng.normal(size=1000)
        keep = x > 0.5
        close(need(ns, "compact")(x, keep), x[keep], what="sıkıştırma")
        k = rng.integers(0, 2 ** 16, 5000)
        close(need(ns, "radix_sort")(k), np.sort(k), what="taban sıralaması")
        a = rng.uniform(0.5, 1.0, 300)
        b = rng.normal(size=300)
        ref = np.zeros(300)
        acc = 0.0
        for i in range(300):
            acc = a[i] * acc + b[i]
            ref[i] = acc
        close(need(ns, "linear_recurrence")(a, b), ref, atol=1e-9, what="doğrusal özyineleme")
        R_.note("Taban sıralamasının her geçişi bir 'bölme' (split) işlemidir: 0'lar sırasını koruyarak öne, 1'ler arkaya — iki dışlayıcı tarama. 16 bit = 16 tarama; GPU'larda 4–8 bitlik basamaklarla sayma sıralaması kullanılır (x3).")
    R_.test("vektörel taramalar ve uygulamalar", t)
