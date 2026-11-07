import numpy as np
from labcheck import close, max_loops, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("en fazla iki seviye döngü", lambda: max_loops(src, 2, func="grads_ckpt"))
    R.test("torch kullanılmıyor", lambda: no_imports(src, {"torch"}))
    rng = np.random.default_rng(0)
    Ws = [rng.normal(size=(6, 6)) / 2.5 for _ in range(10)]
    x, gout = rng.normal(size=(3, 6)), rng.normal(size=(3, 6))

    def full():
        G = need(ns, "grads_full")(x, Ws, gout)
        h = 1e-6
        def L(Ws_):
            hh = x
            for W in Ws_:
                hh = np.tanh(hh @ W)
            return (hh * gout).sum()
        for i in (0, 5, 9):
            num = np.zeros_like(Ws[i])
            for a in range(6):
                for b in range(6):
                    Wp = [w.copy() for w in Ws]; Wp[i][a, b] += h
                    Wm = [w.copy() for w in Ws]; Wm[i][a, b] -= h
                    num[a, b] = (L(Wp) - L(Wm)) / (2 * h)
            close(G[i], num, atol=1e-6, what=f"W{i}")
    R.test("grads_full = sayısal türev", full)

    def ck():
        G = need(ns, "grads_full")(x, Ws, gout)
        rows = []
        for k in (1, 2, 3, 4, 10):
            segs = [list(range(s, min(s + k, 10))) for s in range(0, 10, k)]
            want = max(j + len(sg) for j, sg in enumerate(segs))
            Gc, peak = need(ns, "grads_ckpt")(x, Ws, gout, k)
            for a, b in zip(G, Gc):
                close(a, b, what=f"k = {k}")
            ensure(peak == want, f"k = {k}: tepe {peak}, beklenen {want}")
            rows.append([str(k), str(peak)])
        R.table("10 katman", ["k", "tepe (saklı çıktı)"], rows)
    R.test("checkpoint'li gradyanlar aynı, tepe doğru", ck)
