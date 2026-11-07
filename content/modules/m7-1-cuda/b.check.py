import random
from labcheck import close, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("numpy/torch kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))
    Global = need(ns, "Global", "sınıf")

    def va():
        n = 1000
        x, y, out = Global(range(n)), Global([2 * i for i in range(n)]), Global([None] * n)
        need(ns, "launch")(need(ns, "vec_add"), 4, 256, x, y, out, n)
        ensure(out.data == [3 * i for i in range(n)], "sonuç yanlış")
        ensure(x.reads == n, f"x {x.reads} kez okundu; sınır dışı thread'ler okumamalı")
    R.test("vec_add + launch (sınır kontrolü)", va)
    rng = random.Random(0)

    def mm(fn, *extra):
        n = 8
        a = [rng.uniform(-1, 1) for _ in range(n * n)]; b = [rng.uniform(-1, 1) for _ in range(n * n)]
        A, B = Global(a), Global(b)
        C = fn(A, B, n, *extra)
        ref = [sum(a[i * n + k] * b[k * n + j] for k in range(n)) for i in range(n) for j in range(n)]
        close(C, ref)
        return A.reads + B.reads

    def naive():
        r = mm(need(ns, "matmul_naive"))
        ensure(r == 2 * 8 ** 3, f"{r} okuma; 2n³ = 1024 bekleniyordu")
    R.test("matmul_naive: 2n³ okuma", naive)

    def tiled():
        rows = [["naif", str(2 * 8 ** 3)]]
        for T in (2, 4, 8):
            r = mm(need(ns, "matmul_tiled"), T)
            ensure(r == 2 * 8 ** 3 // T, f"T = {T}: {r} okuma; 2n³/T = {2 * 8 ** 3 // T} bekleniyordu")
            rows.append([f"tile {T}", str(r)])
        R.table("n = 8, global okuma", ["yöntem", "okuma"], rows)
    R.test("matmul_tiled: 2n³/T okuma", tiled)
