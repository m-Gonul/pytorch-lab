import numpy as np
from labcheck import close, max_loops, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("ce_forward/ce_backward döngüsüz", lambda: (max_loops(src, 0, func="ce_forward"), max_loops(src, 0, func="ce_backward")))
    R.test("dir_gradcheck: en fazla tek döngü", lambda: max_loops(src, 1, func="dir_gradcheck"))
    R.test("torch kullanılmıyor", lambda: no_imports(src, {"torch"}))
    rng = np.random.default_rng(0)
    Z, y = rng.normal(size=(8, 5)), rng.integers(0, 5, 8)

    def fw():
        loss, cache = need(ns, "ce_forward")(Z, y)
        p = np.exp(Z) / np.exp(Z).sum(1, keepdims=True)
        close(loss, -np.log(p[np.arange(8), y]).mean())
        ensure(len(cache) == 2, "cache yalnız (olasılıklar, y) olmalı")
        l2, _ = ns["ce_forward"](Z + 1000, y)
        close(l2, loss, what="logit'ler + 1000 (kararlılık)")
    R.test("ce_forward (kararlı)", fw)

    def bw():
        _, cache = need(ns, "ce_forward")(Z, y)
        d = need(ns, "ce_backward")(cache)
        h = 1e-6; num = np.zeros_like(Z)
        for i in range(8):
            for j in range(5):
                Zp = Z.copy(); Zp[i, j] += h; Zm = Z.copy(); Zm[i, j] -= h
                num[i, j] = (ns["ce_forward"](Zp, y)[0] - ns["ce_forward"](Zm, y)[0]) / (2 * h)
        close(d, num, atol=1e-7)
    R.test("ce_backward = sayısal türev", bw)

    def dg():
        f = lambda z: ns["ce_forward"](z, y)[0]
        good = lambda z: ns["ce_backward"](ns["ce_forward"](z, y)[1])
        bad = lambda z: good(z) * 1.01
        e1 = need(ns, "dir_gradcheck")(f, good, Z, np.random.default_rng(1))
        e2 = ns["dir_gradcheck"](f, bad, Z, np.random.default_rng(1))
        ensure(e1 < 1e-5, f"doğru gradyanda hata {e1:.2e}")
        ensure(e2 > 5e-3, f"%1 hatalı gradyanı yakalamalı (hata {e2:.2e})")
        return f"doğru gradyan {e1:.1e}, %1 bozuk gradyan {e2:.1e}"
    R.test("dir_gradcheck hatayı yakalıyor", dg)
