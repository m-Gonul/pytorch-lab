import numpy as np
from labcheck import close, no_loops, no_imports, need, ensure


def py_merge(ids, a, b, new):
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == a and ids[i + 1] == b:
            out.append(new); i += 2
        else:
            out.append(ids[i]); i += 1
    return out


def check(ns, R):
    src = ns["__source__"]
    R.test("döngü yok", lambda: no_loops(src))
    R.test("torch kullanılmıyor", lambda: no_imports(src, {"torch"}))
    rng = np.random.default_rng(0)

    def eb():
        close(need(ns, "embed_backward")(np.array([2, 0, 2]), np.array([[1.0, 1], [2, 2], [3, 3]]), 4), [[2, 2], [0, 0], [4, 4], [0, 0]])
        ids = rng.integers(0, 10, 200); dout = rng.normal(size=(200, 3))
        ref = np.zeros((10, 3))
        for i, d in zip(ids, dout):
            ref[i] += d
        close(need(ns, "embed_backward")(ids, dout, 10), ref, what="200 id")
    R.test("embed_backward: tekrarlı id'ler toplanıyor", eb)

    def pcn():
        ids = rng.integers(0, 6, 300)
        C = need(ns, "pair_counts_np")(ids, 6)
        ref = np.zeros((6, 6), int)
        for a, b in zip(ids[:-1], ids[1:]):
            ref[a, b] += 1
        ensure(np.array_equal(C, ref), "çift sayım matrisi farklı")
    R.test("pair_counts_np", pcn)

    def mg():
        ensure(list(need(ns, "merge_ids")(np.array([0, 1, 2, 0, 1, 1, 0, 1]), 0, 1, 9)) == [9, 2, 9, 1, 9], "elle örnek")
        for _ in range(5):
            ids = rng.integers(0, 4, 100)
            ensure(list(need(ns, "merge_ids")(ids, 1, 2, 7)) == py_merge(list(ids), 1, 2, 7), "rastgele dizi")
    R.test("merge_ids (a ≠ b)", mg)
