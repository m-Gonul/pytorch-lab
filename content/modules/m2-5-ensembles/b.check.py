import math
import random
from labcheck import close, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("numpy/torch/sklearn kullanılmıyor", lambda: no_imports(src, {"numpy", "torch", "sklearn"}))

    def st():
        t, l, r = need(ns, "fit_stump")([1, 2, 3, 4], [-1, -1, 1, 1])
        close([t, l, r], [2.5, -1, 1])
        t2, l2, r2 = need(ns, "fit_stump")([3, 3, 3], [1.0, 2.0, 3.0])
        ensure(t2 is None, "tek farklı x → None"); close([l2, r2], [2, 2])
    R.test("fit_stump: elle örnek, bölünemez durum", st)

    def hand():
        m = need(ns, "gbm_fit")([1, 2, 3, 4], [1, 1, 3, 3], 2, 0.5)
        close([need(ns, "gbm_predict")(m, x) for x in [1, 2, 3, 4]], [1.25, 1.25, 2.75, 2.75])
    R.test("2 tur: [1.25, 1.25, 2.75, 2.75]", hand)

    def curve():
        rng = random.Random(0)
        xs = [rng.uniform(0, 6) for _ in range(80)]
        ys = [math.sin(x) + rng.gauss(0, 0.1) for x in xs]
        mses = []
        for rounds in (1, 10, 100):
            m = need(ns, "gbm_fit")(xs, ys, rounds, 0.3)
            mses.append(sum((need(ns, "gbm_predict")(m, x) - y) ** 2 for x, y in zip(xs, ys)) / 80)
        ensure(mses[0] > mses[1] > mses[2] and mses[2] < 0.02, f"eğitim MSE'leri {mses}")
        return "eğitim MSE (1, 10, 100 tur): " + ", ".join(f"{v:.4f}" for v in mses)
    R.test("sin(x)'e uyum: MSE turlarla azalıyor", curve)
