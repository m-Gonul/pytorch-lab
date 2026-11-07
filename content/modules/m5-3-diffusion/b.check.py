import math
import random
from labcheck import close, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("numpy/torch kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))
    R.test("alpha_bars", lambda: close(need(ns, "alpha_bars")([0.1, 0.2, 0.5]), [0.9, 0.72, 0.36]))
    ab = [0.9, 0.72]
    R.test("q_sample (elle örnek)", lambda: close(need(ns, "q_sample")(2.0, 1, 0.5, ab), 1.961632, atol=1e-6))

    def step():
        x2 = math.sqrt(0.72) * 2 + math.sqrt(0.28) * 0.5
        close(need(ns, "ddpm_step")(x2, 1, 0.5, [0.1, 0.2], ab, 0.0), 1.981884, atol=1e-6, what="z = 0")
        close(ns["ddpm_step"](x2, 1, 0.5, [0.1, 0.2], ab, 1.0), 1.981884 + math.sqrt(0.1 / 0.28 * 0.2), atol=1e-6, what="z = 1")
        close(ns["ddpm_step"](1.0, 0, 0.3, [0.1, 0.2], ab, 5.0), (1 - 0.1 / math.sqrt(0.1) * 0.3) / math.sqrt(0.9), atol=1e-6, what="t = 0: gürültü yok")
    R.test("ddpm_step", step)

    def orc():
        rng = random.Random(3)
        abar = [0.3]
        xs = [math.sqrt(0.3) * 1.5 + math.sqrt(0.7) * 0 for _ in range(1)]
        pairs = []
        for _ in range(40000):
            x0 = rng.gauss(1.5, 0.8); e = rng.gauss(0, 1)
            pairs.append((math.sqrt(0.3) * x0 + math.sqrt(0.7) * e, e))
        xs = [p[0] for p in pairs]; es = [p[1] for p in pairs]
        mx = sum(xs) / len(xs)
        slope = sum((x - mx) * e for x, e in zip(xs, es)) / sum((x - mx) ** 2 for x in xs)
        pred = need(ns, "oracle_eps")(mx + 1.0, 0, abar, 1.5, 0.8) - ns["oracle_eps"](mx, 0, abar, 1.5, 0.8)
        ensure(abs(pred - slope) < 0.02, f"kâhinin eğimi {pred:.4f}, regresyon eğimi {slope:.4f}")
        ensure(abs(ns["oracle_eps"](math.sqrt(0.3) * 1.5, 0, abar, 1.5, 0.8)) < 1e-12, "ortalamada tahmin 0 olmalı")
    R.test("oracle_eps = E[ε | x_t] (regresyonla)", orc)

    def smp():
        T = 100
        betas = [1e-4 + (0.1 - 1e-4) * i / (T - 1) for i in range(T)]
        abar = need(ns, "alpha_bars")(betas)
        xs = need(ns, "sample")(lambda x, t: ns["oracle_eps"](x, t, abar, 3.0, 0.5), betas, 3000, random.Random(0))
        m = sum(xs) / len(xs); s = (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5
        ensure(abs(m - 3) < 0.06 and abs(s - 0.5) < 0.05, f"ortalama {m:.3f}, std {s:.3f}; beklenen 3, 0.5")
        return f"3000 örnek: ortalama {m:.3f}, std {s:.3f} (hedef 3, 0.5)"
    R.test("kâhinle örnekleme N(3, 0.5²) veriyor", smp)
