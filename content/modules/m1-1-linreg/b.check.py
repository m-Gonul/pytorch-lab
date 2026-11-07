import random
from labcheck import close, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("numpy/torch kullanılmıyor", lambda: no_imports(src, {"numpy", "torch", "pandas"}))

    def step():
        out = need(ns, "gd_step")([1, 2, 3], [2, 4, 5], 0.0, 0.0, 0.1)
        ensure(isinstance(out, (tuple, list)) and len(out) == 3, "(w, b, kayıp) üçlüsü döndürmeli")
        close(out, [5 / 3, 11 / 15, 15], what="(w, b, kayıp)")
    R.test("elle adım: (1.667, 0.733, 15)", step)

    def cf():
        close(need(ns, "closed_form")([1, 2, 3], [2, 4, 5]), [1.5, 2 / 3])
        rng = random.Random(4)
        xs = [rng.uniform(-3, 3) for _ in range(30)]
        ys = [0.7 * x - 1.2 + rng.gauss(0, 0.3) for x in xs]
        n = len(xs); xm = sum(xs) / n; ym = sum(ys) / n
        w = sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / sum((x - xm) ** 2 for x in xs)
        close(need(ns, "closed_form")(xs, ys), [w, ym - w * xm], what="30 rastgele nokta")
    R.test("closed_form: elle örnek ve 30 rastgele nokta", cf)

    def fit_conv():
        close(need(ns, "fit")([1, 2, 3], [2, 4, 5]), [1.5, 2 / 3], rtol=0, atol=1e-4, what="fit sonucu")
        return "2000 adımda kapalı form ile 4 basamak eşleşti"
    R.test("fit kapalı forma yakınsıyor", fit_conv)

    def loss_down():
        w = b = 0.0
        prev = float("inf")
        for _ in range(50):
            w, b, L = need(ns, "gd_step")([1, 2, 3], [2, 4, 5], w, b, 0.05)
            ensure(L <= prev + 1e-12, f"kayıp arttı: {prev:.4f} → {L:.4f}")
            prev = L
    R.test("küçük lr ile kayıp hiç artmıyor", loss_down)
