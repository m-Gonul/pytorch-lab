import numpy as np
from labcheck import close, max_loops, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("load_tile döngüsüz", lambda: max_loops(src, 0, func="load_tile"))
    R.test("matmul_program: yalnız K döngüsü", lambda: max_loops(src, 1, func="matmul_program"))
    R.test("torch kullanılmıyor", lambda: no_imports(src, {"torch"}))
    M = np.arange(20.0).reshape(4, 5)

    def lt():
        close(need(ns, "load_tile")(M, 2, 3, 4, 4), [[13, 14, 0, 0], [18, 19, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    R.test("load_tile: kenarda sıfır dolgu", lt)
    rng = np.random.default_rng(0)

    def mm():
        for (m, k, n), (bm, bn, bk) in (((70, 50, 40), (16, 16, 16)), ((33, 17, 9), (8, 4, 16)), ((16, 16, 16), (16, 16, 16))):
            A, B = rng.normal(size=(m, k)), rng.normal(size=(k, n))
            close(need(ns, "matmul")(A, B, bm, bn, bk), A @ B, what=f"{m}×{k}×{n}, tile {bm}/{bn}/{bk}")
    R.test("matmul: tek sayılı şekiller", mm)

    def calls():
        cnt = [0]
        orig = ns["matmul_program"]

        def spy(*a):
            cnt[0] += 1
            return orig(*a)
        ns["matmul_program"] = spy
        g = ns["matmul"].__globals__
        g["matmul_program"] = spy
        try:
            ns["matmul"](rng.normal(size=(70, 50)), rng.normal(size=(50, 40)), 16, 16, 16)
        finally:
            g["matmul_program"] = orig
            ns["matmul_program"] = orig
        ensure(cnt[0] == 15, f"{cnt[0]} program; 5 × 3 = 15 bekleniyordu")
    R.test("grid: 15 program", calls)
