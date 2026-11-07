import numpy as np
from labcheck import close, max_loops, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("adam_update döngüsüz", lambda: max_loops(src, 0, func="adam_update"))
    R.test("zero1_step: yalnız işçi döngüsü", lambda: max_loops(src, 1, func="zero1_step"))
    R.test("torch kullanılmıyor", lambda: no_imports(src, {"torch"}))
    rng = np.random.default_rng(0)
    X, y, w = rng.normal(size=(64, 6)), rng.normal(size=64), rng.normal(size=6)

    def dp():
        for K in (1, 2, 4, 8):
            gs = need(ns, "worker_grads")(w, X, y, K)
            ensure(len(gs) == K, f"K = {K} gradyan")
            close(need(ns, "allreduce_mean")(gs), 2 * X.T @ (X @ w - y) / 64, what=f"K = {K}")
    R.test("veri paralel gradyan = tam batch gradyanı", dp)

    def ad():
        g = rng.normal(size=6); m = rng.normal(size=6) * 0.1; v = rng.random(6) * 0.1
        w2, m2, v2 = need(ns, "adam_update")(w, g, m, v, 3)
        mm = 0.9 * m + 0.1 * g; vv = 0.999 * v + 0.001 * g * g
        close(w2, w - 1e-2 * (mm / (1 - 0.9 ** 3)) / (np.sqrt(vv / (1 - 0.999 ** 3)) + 1e-8))
    R.test("adam_update (bias düzeltmeli)", ad)

    def z1():
        W = rng.normal(size=10)
        Wf, m, v = W.copy(), np.zeros(10), np.zeros(10)
        K = 3
        states = [(np.zeros(len(s)), np.zeros(len(s))) for s in np.array_split(W, K)]
        Wz = W.copy()
        for t in range(1, 21):
            g = np.sin(np.arange(10) * t) + Wz * 0.1
            gf = np.sin(np.arange(10) * t) + Wf * 0.1
            Wf, m, v = need(ns, "adam_update")(Wf, gf, m, v, t)
            Wz, states = need(ns, "zero1_step")(Wz, g, states, t)
        close(Wz, Wf)
        sizes = [len(s[0]) for s in states]
        ensure(sizes == [4, 3, 3], f"işçi başına durum boyutu {sizes}")
        return f"20 adım, 3 işçi: aynı ağırlıklar; durum boyutları {sizes} (toplam 10)"
    R.test("ZeRO-1 = tek işçili Adam", z1)
