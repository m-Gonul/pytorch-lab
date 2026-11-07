import random
from labcheck import close, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("numpy/torch kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))

    def mb():
        f = need(ns, "minibatches")
        bs = f(103, 10, random.Random(1))
        ensure(len(bs) == 11 and all(len(b) == 10 for b in bs[:-1]) and len(bs[-1]) == 3, f"batch boyları {[len(b) for b in bs]}")
        flat = [i for b in bs for i in b]
        ensure(sorted(flat) == list(range(103)), "her örnek tam bir kez görülmeli")
        ensure(flat != list(range(103)), "sıra karıştırılmamış")
        ensure(f(20, 5, random.Random(7)) == f(20, 5, random.Random(7)), "aynı rng tohumu aynı batch'leri vermeli")
    R.test("minibatches: karıştırılmış, örtüşmeyen, eksiksiz", mb)

    xs, ys = [1.0, 2.0, 3.0], [2.0, 4.0, 5.0]

    def full():
        w, b = need(ns, "sgd_epoch")(xs, ys, 0.0, 0.0, 0.1, 3, random.Random(0))
        close([w, b], [5 / 3, 11 / 15], what="batch_size = n → tek GD adımı")
    R.test("batch_size = n: M1.1'deki tam GD adımı", full)

    def conv():
        rng = random.Random(0)
        w = b = 0.0
        for _ in range(3000):
            w, b = need(ns, "sgd_epoch")(xs, ys, w, b, 0.01, 1, rng)
        close([w, b], [1.5, 2 / 3], atol=0.05, rtol=0, what="SGD (batch 1) sonucu")
        return f"SGD 3000 epoch: w = {w:.3f}, b = {b:.3f} (kapalı form 1.5, 0.667; gürültü yüzünden tam oturmaz)"
    R.test("batch_size = 1 kapalı forma yaklaşıyor", conv)

    def mom():
        p = need(ns, "momentum_path")(lambda w: 2 * (w - 3), 0.0, 0.25, 0.5, 4)
        ensure(len(p) == 4, "4 değer bekleniyordu")
        close(p[:2], [1.5, 3.0])
        p0 = need(ns, "momentum_path")(lambda w: 2 * (w - 3), 0.0, 0.25, 0.0, 2)
        close(p0, [1.5, 2.25], what="β = 0 düz GD olmalı")
    R.test("momentum_path: [1.5, 3.0, …], β = 0 düz GD", mom)
