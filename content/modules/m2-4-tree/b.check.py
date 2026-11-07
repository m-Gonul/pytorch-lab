import random
from labcheck import close, no_imports, need, ensure


def ref_best(X, y):
    n, d = len(X), len(X[0])
    G = lambda s: 1 - sum((s.count(c) / len(s)) ** 2 for c in set(s)) if s else 0.0
    best = None
    for j in range(d):
        vals = sorted(set(r[j] for r in X))
        for a, b in zip(vals, vals[1:]):
            t = (a + b) / 2
            L = [y[i] for i in range(n) if X[i][j] <= t]; R_ = [y[i] for i in range(n) if X[i][j] > t]
            s = len(L) / n * G(L) + len(R_) / n * G(R_)
            if best is None or s < best[2] - 1e-12:
                best = (j, t, s)
    return best


def check(ns, R):
    src = ns["__source__"]
    R.test("numpy/torch/sklearn kullanılmıyor", lambda: no_imports(src, {"numpy", "torch", "sklearn"}))
    R.test("gini", lambda: close([need(ns, "gini")([0, 0, 1, 1]), need(ns, "gini")([1, 1, 1, 0]), need(ns, "gini")([2, 2])], [0.5, 0.375, 0.0]))
    X1, y1 = [[1], [2], [3], [4], [5], [6]], [0, 0, 1, 1, 1, 0]

    def hand():
        j, t, s = need(ns, "best_split")(X1, y1)
        ensure(j == 0, "özellik 0"); close([t, s], [2.5, 0.25])
        ensure(need(ns, "best_split")([[1], [1]], [0, 1]) is None, "tek farklı değer → None")
    R.test("best_split: elle örnek (2.5, 0.25), bölünemezse None", hand)
    rng = random.Random(0)
    X = [[round(rng.uniform(0, 10), 3) for _ in range(3)] for _ in range(40)]
    y = [int(x[1] > 4) ^ int(x[2] > 7) for x in X]

    def rnd():
        got, want = need(ns, "best_split")(X, y), ref_best(X, y)
        ensure(got[0] == want[0], f"özellik {got[0]}, {want[0]} bekleniyordu"); close(got[1:], want[1:])
    R.test("rastgele 3 özellik: referansla aynı bölünme", rnd)

    def tree():
        T = need(ns, "build_tree")(X, y, 4)
        acc = sum(need(ns, "predict_tree")(T, x) == t for x, t in zip(X, y)) / 40
        ensure(acc == 1.0, f"derinlik 4 ile eğitim doğruluğu {acc}, 1.0 bekleniyordu (XOR yapısı)")
        T1 = need(ns, "build_tree")(X, y, 1)
        ensure("leaf" in T1["left"] and "leaf" in T1["right"], "max_depth = 1 tek bölünme olmalı")
        T0 = need(ns, "build_tree")(X1, y1, 0)
        ensure(T0 == {"leaf": 0}, "max_depth = 0: çoğunluk yaprağı (eşitlikte küçük etiket)")
        return "derinlik 4 ağaç XOR benzeri sınırı tam ayırıyor"
    R.test("build_tree + predict_tree", tree)
