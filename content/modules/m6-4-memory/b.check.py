import math
from labcheck import close, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("numpy/torch kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))
    cc = lambda *a: tuple(need(ns, "checkpoint_cost")(*a))
    R.test("16 katman, k = 4", lambda: ensure(cc([1] * 16, 4) == (7, 12), f"{cc([1] * 16, 4)}"))
    R.test("k = 1 (checkpoint yok)", lambda: ensure(cc([1] * 16, 1) == (16, 0), f"{cc([1] * 16, 1)}"))
    R.test("k = n (yalnız son çıktı)", lambda: ensure(cc([1] * 16, 16) == (16, 15), f"{cc([1] * 16, 16)}"))
    R.test("eşit olmayan son segment (n = 10, k = 4)", lambda: ensure(cc([1] * 10, 4) == (5, 7), f"{cc([1] * 10, 4)}"))
    R.test("farklı boyutlar", lambda: ensure(cc([4, 1, 1, 1, 8, 1], 2) == (11, 3), f"{cc([4, 1, 1, 1, 8, 1], 2)}"))

    def bk():
        rows = []
        for n in (16, 64, 100):
            k = need(ns, "best_k")([1] * n)
            p = cc([1] * n, k)[0]
            rows.append([str(n), str(k), str(p), f"{2 * math.sqrt(n):.1f}"])
            ensure(abs(k - math.sqrt(n)) <= 2, f"n = {n} için k = {k}; ≈ √n bekleniyordu")
        R.table("En iyi checkpoint aralığı", ["n", "k", "tepe", "2√n"], rows)
    R.test("best_k ≈ √n", bk)
