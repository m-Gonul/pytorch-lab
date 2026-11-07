import math
import random
from labcheck import close, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("numpy/torch kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))

    def ls():
        mem = [10, 20, 30]
        ensure(need(ns, "load")(mem, [0, 2, 5], [True, True, False], other=-1) == [10, 30, -1], "maskeli load")
        need(ns, "store")(mem, [1, 7], [99, 77], [True, False])
        ensure(mem == [10, 99, 30], "maskeli store dizinin dışına yazmamalı")
    R.test("load/store maskeleri", ls)

    def add():
        n = 1000
        x, y, out = list(range(n)), [0.5] * n, [None] * n
        need(ns, "launch")(need(ns, "add_program"), 4, x, y, out, n, 256)
        ensure(out == [i + 0.5 for i in range(n)], "sonuç yanlış")
        out2 = [None] * (n + 5)
        ns["launch"](ns["add_program"], 4, x, y, out2, n, 256)
        ensure(out2[n:] == [None] * 5, "maskeli program n'yi aşmamalı")
    R.test("add_program", add)

    def sm():
        rng = random.Random(0)
        rows, cols = 5, 13
        X = [rng.uniform(-30, -10) for _ in range(rows * cols)]
        Y = [None] * (rows * cols)
        need(ns, "launch")(need(ns, "softmax_program"), rows, X, Y, cols, 16)
        for r in range(rows):
            row = X[r * cols:(r + 1) * cols]
            m = max(row); e = [math.exp(v - m) for v in row]
            close(Y[r * cols:(r + 1) * cols], [v / sum(e) for v in e], what=f"satır {r}")
    R.test("softmax_program (tamamen negatif satırlar, BLOCK > n_cols)", sm)
