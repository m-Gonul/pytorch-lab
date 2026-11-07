import math
import random
from labcheck import close, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("numpy/torch kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))

    def hand():
        out, cache = need(ns, "bn_forward")([[1.0], [2.0], [3.0]], [2.0], [1.0], eps=0.0)
        close([r[0] for r in out], [1 - 2 * 1.5 ** 0.5, 1, 1 + 2 * 1.5 ** 0.5])
        dX, dg, db = need(ns, "bn_backward")([[1.0], [0.0], [0.0]], cache)
        close([r[0] for r in dX], [0.408248, -0.816497, 0.408248], atol=1e-6); close(dg, [-1.224745], atol=1e-6); close(db, [1])
    R.test("elle örnek", hand)

    def num():
        rng = random.Random(0)
        X = [[rng.gauss(0, 2) for _ in range(3)] for _ in range(5)]
        g = [rng.uniform(0.5, 2) for _ in range(3)]; b = [rng.gauss(0, 1) for _ in range(3)]
        W = [[rng.gauss(0, 1) for _ in range(3)] for _ in range(5)]
        L = lambda X_: sum(o * w for ro, rw in zip(need(ns, "bn_forward")(X_, g, b)[0], W) for o, w in zip(ro, rw))
        _, cache = need(ns, "bn_forward")(X, g, b)
        dX, _, _ = need(ns, "bn_backward")(W, cache)
        h = 1e-6
        for i in range(5):
            for j in range(3):
                Xp = [r[:] for r in X]; Xm = [r[:] for r in X]; Xp[i][j] += h; Xm[i][j] -= h
                close(dX[i][j], (L(Xp) - L(Xm)) / (2 * h), rtol=1e-4, atol=1e-6, what=f"dX[{i}][{j}]")
        ensure(all(abs(sum(dX[i][j] for i in range(5))) < 1e-9 for j in range(3)), "Σᵢ dXᵢ sıfır olmalı")
    R.test("bn_backward sayısal gradyanla aynı, sütun toplamları 0", num)
    R.test("rms_norm", lambda: close(need(ns, "rms_norm")([3.0, 4.0], [1.0, 2.0], eps=0.0), [3 / 12.5 ** 0.5, 8 / 12.5 ** 0.5]))
