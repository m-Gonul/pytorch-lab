import numpy as np
from labcheck import close, no_imports, need, ensure


def dense(n, edges):
    A = np.eye(n)
    for i, j in edges:
        A[i, j] = A[j, i] = 1
    d = A.sum(1)
    return A / np.sqrt(d[:, None] * d[None, :])


def check(ns, R):
    src = ns["__source__"]
    R.test("numpy/torch kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))

    def w():
        W = need(ns, "gcn_weights")(3, [(0, 1), (1, 2)])
        ensure(len(W) == 7, f"3 öz-döngü + 2 kenar × 2 yön = 7 ağırlık; {len(W)} var")
        close([W[(0, 0)], W[(0, 1)], W[(1, 0)], W[(1, 1)]], [0.5, 1 / 6 ** 0.5, 1 / 6 ** 0.5, 1 / 3])
    R.test("gcn_weights (elle örnek)", w)
    R.test("gcn_layer (elle örnek)", lambda: close(need(ns, "gcn_layer")([[1.0], [2.0], [3.0]], 3, [(0, 1), (1, 2)], [[1.0]]), [[1.316497], [2.29966], [2.316497]], atol=1e-5))

    def rnd():
        rng = np.random.default_rng(0)
        n = 7; edges = [(0, 1), (0, 2), (1, 2), (2, 3), (4, 5)]
        X = rng.normal(size=(n, 3)); W = rng.normal(size=(3, 2))
        close(need(ns, "gcn_layer")(X.tolist(), n, edges, W.tolist()), dense(n, edges) @ X @ W, what="izole düğüm 6 dahil")
    R.test("rastgele graf = yoğun Â X W", rnd)

    def mn():
        out = need(ns, "mean_neighbors")([[1.0, 0], [2.0, 1], [3.0, 2], [9.0, 9]], 4, [(0, 1), (1, 2), (0, 2)])
        close(out, [[2.5, 1.5], [2, 1], [1.5, 0.5], [0, 0]])
    R.test("mean_neighbors (izole düğüm → 0)", mn)
