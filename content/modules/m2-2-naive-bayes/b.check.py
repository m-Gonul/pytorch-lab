import math
import random
from labcheck import close, no_imports, need, ensure


def ref_proba(rows, labels, alpha, q):
    classes = sorted(set(labels))
    d = len(rows[0])
    V = [len({r[j] for r in rows}) for j in range(d)]
    sc = {}
    for c in classes:
        sub = [r for r, l in zip(rows, labels) if l == c]
        s = len(sub) / len(rows)
        for j in range(d):
            s *= (sum(r[j] == q[j] for r in sub) + alpha) / (len(sub) + alpha * V[j])
        sc[c] = s
    z = sum(sc.values())
    return {c: v / z for c, v in sc.items()}


def check(ns, R):
    src = ns["__source__"]
    R.test("numpy/torch/pandas kullanılmıyor", lambda: no_imports(src, {"numpy", "torch", "pandas", "sklearn"}))
    rows = [("g", "z"), ("g", "s"), ("y", "s"), ("y", "z"), ("g", "s"), ("y", "s")]
    labels = ["E", "E", "H", "E", "H", "H"]

    def hand():
        p0 = need(ns, "predict_proba")(need(ns, "fit_nb")(rows, labels, alpha=0.0), ("g", "s"))
        p1 = need(ns, "predict_proba")(need(ns, "fit_nb")(rows, labels, alpha=1.0), ("g", "s"))
        close([p0["E"], p0["H"]], [0.4, 0.6], what="α = 0")
        close([p1["E"], p1["H"]], [3 / 7, 4 / 7], what="α = 1")
    R.test("elle örnek: 0.4 (α = 0) ve 0.4286 (α = 1)", hand)

    def zero():
        p = need(ns, "predict_proba")(need(ns, "fit_nb")(rows, labels, alpha=0.0), ("y", "z"))
        close([p["E"], p["H"]], [1.0, 0.0], what="(yağmur, zayıf), α = 0")
        p1 = need(ns, "predict_proba")(need(ns, "fit_nb")(rows, labels, alpha=1.0), ("y", "z"))
        ensure(0 < p1["H"] < 1, "Laplace ile H olasılığı 0 ile 1 arasında olmalı")
        return f"α = 0: P(H) = 0; α = 1: P(H) = {p1['H']:.3f}"
    R.test("sıfır sayım ve Laplace", zero)

    def rnd():
        rng = random.Random(0)
        rows2 = [tuple(rng.choice("abc") for _ in range(4)) for _ in range(60)]
        labs = [("x" if r.count("a") >= 2 else "y") if rng.random() < 0.9 else rng.choice("xy") for r in rows2]
        m = need(ns, "fit_nb")(rows2, labs, alpha=0.5)
        for q in rows2[:10]:
            got = need(ns, "predict_proba")(m, q)
            want = ref_proba(rows2, labs, 0.5, q)
            close([got[c] for c in sorted(want)], [want[c] for c in sorted(want)], rtol=1e-9)
            ensure(need(ns, "predict")(m, q) == max(want, key=want.get), "predict argmax değil")
    R.test("rastgele veri: referansla aynı, predict = argmax", rnd)

    def underflow():
        rows3 = [tuple("ab"[(i >> k) & 1] for k in range(400)) for i in range(8)]
        labs3 = ["p", "q"] * 4
        p = need(ns, "predict_proba")(need(ns, "fit_nb")(rows3, labs3, alpha=1.0), rows3[0])
        ensure(all(math.isfinite(v) for v in p.values()) and abs(sum(p.values()) - 1) < 1e-9, "400 özellikte olasılıklar bozuldu; log uzayında çalış")
    R.test("400 özellikte alt taşma yok (log uzayı)", underflow)
