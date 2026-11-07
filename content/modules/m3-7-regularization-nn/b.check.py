import random
from labcheck import close, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("numpy/torch kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))

    def fwd():
        y, m = need(ns, "dropout_forward")([2, 4, 6, 8], 0.5, random.Random(0))
        r = random.Random(0); want = [1 if r.random() >= 0.5 else 0 for _ in range(4)]
        ensure(list(m) == want, f"maske {m}, {want} bekleniyordu (eleman başına sırayla rng.random())")
        close(y, [v * k / 0.5 for v, k in zip([2, 4, 6, 8], want)])
        y2, m2 = need(ns, "dropout_forward")([2, 4], 0.5, random.Random(0), train=False)
        ensure(list(y2) == [2, 4] and m2 is None, "eval'de kimlik ve mask None")
    R.test("dropout_forward: maske, ölçek, eval", fwd)

    def unb():
        rng = random.Random(1)
        n = 20000
        y, _ = need(ns, "dropout_forward")([3.0] * n, 0.3, rng)
        ensure(abs(sum(y) / n - 3.0) < 0.05, f"ortalama {sum(y) / n:.3f}, 3 bekleniyordu")
        return f"20 000 örnek: ortalama {sum(y) / n:.4f} (girdi 3.0)"
    R.test("beklenen değer korunuyor", unb)
    R.test("dropout_backward", lambda: close(need(ns, "dropout_backward")([1.0, 1.0, 2.0], [1, 0, 1], 0.2), [1.25, 0, 2.5]))

    def es():
        f = need(ns, "early_stopping")
        ensure(tuple(f([0.9, 0.7, 0.6, 0.62, 0.65, 0.61], 2)) == (2, 4), "elle örnek (2, 4)")
        ensure(tuple(f([1.0, 0.9, 0.8], 3)) == (2, 2), "hiç durmazsa stop = son indeks")
        ensure(tuple(f([1.0, 0.995, 0.99, 0.985], 2, min_delta=0.01)) == (0, 2), "min_delta küçük iyileşmeleri saymamalı")
    R.test("early_stopping (min_delta dahil)", es)
