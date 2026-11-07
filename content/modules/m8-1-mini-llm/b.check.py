import math
from labcheck import close, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("numpy/torch kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))

    def lr():
        f = need(ns, "lr_schedule")
        close([f(0, 100, 1100, 1e-3, 1e-4), f(49, 100, 1100, 1e-3, 1e-4), f(99, 100, 1100, 1e-3, 1e-4)], [1e-5, 5e-4, 1e-3])
        close([f(100, 100, 1100, 1e-3, 1e-4), f(600, 100, 1100, 1e-3, 1e-4), f(1100, 100, 1100, 1e-3, 1e-4), f(5000, 100, 1100, 1e-3, 1e-4)], [1e-3, 5.5e-4, 1e-4, 1e-4])
        vals = [f(t, 100, 1100, 1e-3, 1e-4) for t in range(100, 1100)]
        ensure(all(a >= b for a, b in zip(vals, vals[1:])), "warmup'tan sonra monoton azalmalı")
    R.test("lr_schedule", lr)
    R.test("bütçe", lambda: (ensure(need(ns, "train_flops")(1e7, 2e8) == 1.2e16, "6ND"), ensure(need(ns, "chinchilla_tokens")(1e7) == 2e8, "20N")))

    def fmt():
        ml = need(ns, "make_line")
        ensure(ml(7, 45) == "07+45=250\n" and ml(99, 99) == "99+99=891\n" and ml(0, 0) == "00+00=000\n", "biçim")
        ensure(all(len(ml(a, b)) == 10 for a in range(0, 100, 7) for b in range(0, 100, 11)), "her satır 10 karakter")
        pa = need(ns, "parse_answer")
        ensure(all(pa(ml(a, b)[6:9]) == a + b for a in range(100) for b in range(0, 100, 13)), "gidiş-dönüş")
    R.test("make_line / parse_answer", fmt)
