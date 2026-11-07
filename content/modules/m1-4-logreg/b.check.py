import math
import random
from labcheck import close, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("numpy/torch kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))

    def sig():
        s = need(ns, "sigmoid")
        close([s(-1), s(0), s(2)], [1 / (1 + math.e), 0.5, 1 / (1 + math.exp(-2))])
        close([s(-1000), s(1000), s(-745.2)], [0.0, 1.0, 0.0], atol=1e-300)
    R.test("sigmoid doğru ve ±1000'de taşmıyor", sig)

    def bc():
        f = need(ns, "bce")
        close([f(0.5, 1), f(0.25, 0)], [math.log(2), -math.log(0.75)])
        ensure(math.isfinite(f(0.0, 1)) and math.isfinite(f(1.0, 0)), "p = 0 ya da 1'de sonsuz kayıp; kırpma (eps) eksik")
    R.test("bce doğru ve p ∈ {0, 1}'de sonlu", bc)

    def gr():
        out = need(ns, "grad")([[-1.0], [0.0], [2.0]], [0, 1, 1], [1.0], 0.0)
        ensure(isinstance(out, (tuple, list)) and len(out) == 3, "(gw, gb, L) döndürmeli")
        close(out[0], [-0.169116], atol=1e-5, what="gw")
        close(out[1], -0.116754, atol=1e-5, what="gb")
        close(out[2], 0.377779, atol=1e-5, what="L")
    R.test("elle örnek: gw ≈ −0.1691, gb ≈ −0.1168, L ≈ 0.3778", gr)

    def ft():
        rng = random.Random(0)
        X, y = [], []
        for _ in range(60):
            c = rng.random() < 0.5
            X.append([rng.gauss(1.5 if c else -1.5, 0.8), rng.gauss(1.0 if c else -1.0, 0.8)])
            y.append(int(c))
        w, b = need(ns, "fit")(X, y, lr=0.5, steps=300)
        acc = sum((sum(wi * xi for wi, xi in zip(w, x)) + b > 0) == bool(t) for x, t in zip(X, y)) / len(y)
        ensure(acc >= 0.9, f"eğitim doğruluğu {acc:.2f}, ≥ 0.90 bekleniyordu")
        return f"2 boyutlu iki küme: doğruluk {acc:.2f}"
    R.test("fit iki kümeyi ayırıyor", ft)
