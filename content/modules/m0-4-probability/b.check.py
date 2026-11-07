import math
import random
from labcheck import close, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("numpy/torch/statistics kullanılmıyor", lambda: no_imports(src, {"numpy", "torch", "statistics", "pandas"}))

    def mv():
        close([need(ns, "mean")([2, 4, 6]), need(ns, "var")([2, 4, 6]), need(ns, "var")([2, 4, 6], ddof=1)], [4, 8 / 3, 4])
        rng = random.Random(0)
        xs = [rng.gauss(3, 2) for _ in range(50)]
        m = sum(xs) / 50
        close(need(ns, "var")(xs), sum((v - m) ** 2 for v in xs) / 50, what="50 örnek MLE varyansı")
    R.test("mean, var (ddof = 0 ve 1)", mv)

    def ll():
        f = need(ns, "bernoulli_ll")
        close(f([1, 0, 1, 1], 0.75), 3 * math.log(0.75) + math.log(0.25))
        ensure(f([1, 1], 0.0) == -math.inf, "p = 0 iken tura gelmişse −inf bekleniyordu")
        ensure(f([0, 0], 1.0) == -math.inf, "p = 1 iken yazı gelmişse −inf bekleniyordu")
        close(f([1, 1], 1.0), 0.0, what="p = 1, hep tura")
    R.test("bernoulli_ll ve p ∈ {0, 1} uç durumları", ll)

    def underflow():
        rng = random.Random(1)
        ys = [int(rng.random() < 0.6) for _ in range(2000)]
        prod = math.prod(0.6 if y else 0.4 for y in ys)
        v = need(ns, "bernoulli_ll")(ys, 0.6)
        ensure(prod == 0.0 and math.isfinite(v), "çarpım 0'a düşmeli, log toplamı sonlu kalmalı")
        return f"2000 atış: çarpım = {prod}, log-likelihood = {v:.1f}"
    R.test("log toplamı alt taşmadan korur", underflow)

    def cat():
        close(need(ns, "categorical_mle")([1, 3, 3, 6, 3, 1], 6), [2 / 6, 0, 3 / 6, 0, 0, 1 / 6])
        p = need(ns, "categorical_mle")([2] * 5, 3)
        close(p, [0, 1, 0], what="hep 2")
    R.test("categorical_mle", cat)
