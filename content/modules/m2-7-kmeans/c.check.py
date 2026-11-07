import numpy as np
from labcheck import close, max_loops, no_imports, need, ensure


def ref(X, C, iters):
    C = C.copy()
    for _ in range(iters):
        lab = ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)
        for k in range(len(C)):
            if (lab == k).any():
                C[k] = X[lab == k].mean(0)
    D = ((X[:, None] - C[None]) ** 2).sum(-1)
    lab = D.argmin(1)
    return C, lab, D[np.arange(len(X)), lab].sum()


def check(ns, R):
    src = ns["__source__"]
    R.test("kmeans_np: yalnız iterasyon döngüsü", lambda: max_loops(src, 1, func="kmeans_np"))
    R.test("torch/sklearn kullanılmıyor", lambda: no_imports(src, {"torch", "sklearn"}))
    rng = np.random.default_rng(0)
    X = np.concatenate([rng.normal(c, 0.6, (40, 2)) for c in ([0, 0], [5, 5], [0, 5], [5, 0])])

    def km():
        C0 = X[[0, 1, 2, 3]].copy()
        C, lab, J = need(ns, "kmeans_np")(X, C0, 8)
        Cr, labr, Jr = ref(X, C0, 8)
        close(C, Cr); ensure((np.asarray(lab) == labr).all(), "etiketler farklı"); close(J, Jr)
        C, lab, J = need(ns, "kmeans_np")(np.array([[0., 0], [1, 0], [4, 4], [5, 4]]), np.array([[0., 0], [1, 0]]), 2)
        close(J, 1.0, what="elle örnek")
    R.test("kmeans_np referansla aynı (4 kümenin 1. kümeden başlatılması)", km)

    def empty():
        C, lab, J = need(ns, "kmeans_np")(np.array([[0., 0], [1, 1]]), np.array([[0.5, 0.5], [100., 100]]), 3)
        close(C, [[0.5, 0.5], [100, 100]], what="boş küme eski merkezini korur")
    R.test("boş küme", empty)

    def pp():
        C = need(ns, "kmeanspp")(X, 4, np.random.default_rng(3))
        r = np.random.default_rng(3)
        want = [X[r.integers(len(X))]]
        for _ in range(3):
            D2 = ((X[:, None] - np.array(want)[None]) ** 2).sum(-1).min(1)
            want.append(X[r.choice(len(X), p=D2 / D2.sum())])
        close(C, np.array(want))
        Jpp = np.mean([need(ns, "kmeans_np")(X, need(ns, "kmeanspp")(X, 4, np.random.default_rng(s)), 10)[2] for s in range(10)])
        Jrnd = np.mean([need(ns, "kmeans_np")(X, X[np.random.default_rng(s).choice(len(X), 4, replace=False)], 10)[2] for s in range(10)])
        return f"10 tohum ortalaması inertia: k-means++ {Jpp:.1f}, rastgele {Jrnd:.1f}"
    R.test("kmeanspp: belirtilen rng dizisi", pp)
