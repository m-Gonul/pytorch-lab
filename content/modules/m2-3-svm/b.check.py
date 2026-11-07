import random
from labcheck import close, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("numpy/torch kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))

    def hand():
        L, gw, gb = need(ns, "hinge_grad")([[2, 1], [0, 1], [1, 1]], [1, -1, 1], [1.0, -1.0], 0.0, 0.0)
        close([L, gw[0], gw[1], gb], [1 / 3, -1 / 3, -1 / 3, -1 / 3])
        L2, gw2, _ = need(ns, "hinge_grad")([[2, 1]], [1], [1.0, -1.0], 0.0, 0.5)
        close([L2, gw2[0], gw2[1]], [1.0, 1.0, -1.0], what="yalnız ceza (λ = 0.5)")
    R.test("hinge_grad: elle örnek ve L2 cezası", hand)
    R.test("multiclass_hinge = 2.9", lambda: close(need(ns, "multiclass_hinge")([3.2, 5.1, -1.7], 0), 2.9))

    def fit():
        rng = random.Random(0)
        X, y = [], []
        for _ in range(80):
            c = rng.choice([-1, 1])
            X.append([rng.gauss(1.5 * c, 0.8), rng.gauss(-1.0 * c, 0.8)]); y.append(c)
        w, b = need(ns, "fit_svm")(X, y, lam=0.01, lr=0.1, steps=500)
        acc = sum((sum(a * b_ for a, b_ in zip(w, x)) + b > 0) == (t > 0) for x, t in zip(X, y)) / 80
        ensure(acc >= 0.9, f"doğruluk {acc:.2f}")
        return f"doğruluk {acc:.3f}, w = [{w[0]:.2f}, {w[1]:.2f}]"
    R.test("fit_svm iki kümeyi ayırıyor", fit)
