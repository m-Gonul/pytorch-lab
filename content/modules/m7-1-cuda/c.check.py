import numpy as np
from labcheck import close, max_loops, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("intensity/attainable döngüsüz", lambda: (max_loops(src, 0, func="intensity"), max_loops(src, 0, func="attainable")))
    R.test("torch kullanılmıyor", lambda: no_imports(src, {"torch"}))
    R.test("intensity", lambda: close(need(ns, "intensity")([1, 10], [12, 4]), [1 / 12, 2.5]))

    def att():
        I = np.array([0.083, 1.0, 12.54, 100.0])
        close(need(ns, "attainable")(19.5e12, 1.555e12, I), np.minimum(19.5e12, 1.555e12 * I))
    R.test("attainable (roofline)", att)

    def oc():
        c = need(ns, "op_costs")(1024)
        ensure(tuple(c["add"]) == (1024, 12 * 1024), "add: n FLOP, 3n·4 bayt")
        ensure(tuple(c["matmul"]) == (2 * 1024 ** 3, 12 * 1024 ** 2), "matmul: 2n³ FLOP, 3n²·4 bayt")
        ensure(tuple(c["softmax"]) == (5 * 1024 ** 2, 8 * 1024 ** 2), "softmax: 5n² FLOP, 2n²·4 bayt")
        ridge = 19.5e12 / 1.555e12
        rows = []
        for name, (f, b) in c.items():
            I = float(ns["intensity"](f, b))
            rows.append([name, f"{I:.3f}", "hesap" if I > ridge else "bellek"])
        R.table(f"n = 1024, A100 fp32 (tepe noktası {ridge:.1f} FLOP/bayt)", ["işlem", "FLOP/bayt", "bağımlı"], rows)
    R.test("op_costs ve sınıflandırma", oc)

    def mb():
        rng = np.random.default_rng(0)
        A, B = rng.normal(size=(12, 8)), rng.normal(size=(8, 20))
        close(need(ns, "matmul_blocked")(A, B, 4), A @ B)
        close(ns["matmul_blocked"](A, B, 5), A @ B, what="T bölmüyor (kenar tile'ları)")
    R.test("matmul_blocked (dikdörtgen, kenar tile'ları)", mb)
