import numpy as np
from labcheck import close, no_loops, no_imports, need, ensure


def rand_graph(rng, n, p):
    A = np.triu(rng.random((n, n)) < p, 1)
    A = A | A.T
    return np.array(np.nonzero(A))


def check(ns, R):
    src = ns["__source__"]
    R.test("döngü yok", lambda: no_loops(src))
    R.test("torch kullanılmıyor", lambda: no_imports(src, {"torch"}))
    rng = np.random.default_rng(0)
    ei = rand_graph(rng, 30, 0.15); X = rng.normal(size=(30, 4))

    def dn():
        A = np.eye(30)
        for s, t in ei.T:
            A[s, t] = 1
        d = A.sum(1)
        close(need(ns, "dense_norm_adj")(ei, 30), A / np.sqrt(np.outer(d, d)))
    R.test("dense_norm_adj", dn)
    R.test("propagate = Â X (scatter-add)", lambda: close(need(ns, "propagate")(ei, X, 30), need(ns, "dense_norm_adj")(ei, 30) @ X))

    def perm():
        p = rng.permutation(30); inv = np.argsort(p)
        ei2 = inv[ei]
        X2 = X[p]
        close(need(ns, "propagate")(ei2, X2, 30), ns["propagate"](ei, X, 30)[p], what="yeniden numaralandırma")
    R.test("permütasyon eşdeğişkenliği", perm)

    def iso():
        out = need(ns, "propagate")(np.zeros((2, 0), int), X[:5], 5)
        close(out, X[:5], what="kenarsız graf: Â = I")
    R.test("kenarsız graf", iso)
