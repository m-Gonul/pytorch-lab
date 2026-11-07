import numpy as np
from labcheck import close, max_loops, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("döngüsüz", lambda: max_loops(src, 0, func="max_subarray"))

    def t():
        ensure(need(ns, "max_subarray")(np.array([-2, 1, -3, 4, -1, 2, 1, -5, 4])) == 6, "klasik örnek")
        ensure(ns["max_subarray"](np.array([-3, -1, -2])) == -1, "hepsi negatif")
        rng = np.random.default_rng(0)
        for _ in range(20):
            x = rng.integers(-10, 10, rng.integers(1, 200))
            ensure(ns["max_subarray"](x) == need(ns, "kadane")(list(x)), "Kadane ile aynı")
        R_.note("İki tarama (toplam ve minimum) — ikisi de birleşmeli, dolayısıyla O(log n) derinlikte paralel. Kadane'nin sıralı özyinelemesi de (4 değerli bir yarı-grup üzerinde) doğrudan taranabilir.")
    R_.test("en büyük alt dizi", t)
