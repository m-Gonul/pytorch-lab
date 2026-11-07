import numpy as np
from labcheck import close, max_loops, no_imports, need, ensure


def ref_pred(D, y, k):
    out = []
    for row in D:
        order = sorted(range(len(row)), key=lambda j: (row[j], j))[:k]
        cnt = {}
        for j in order:
            cnt[int(y[j])] = cnt.get(int(y[j]), 0) + 1
        out.append(min(cnt, key=lambda c: (-cnt[c], c)))
    return out


def check(ns, R):
    src = ns["__source__"]
    R.test("sqdist_one_loop: en fazla tek döngü", lambda: max_loops(src, 1, func="sqdist_one_loop"))
    R.test("sqdist_no_loop: döngü yok", lambda: max_loops(src, 0, func="sqdist_no_loop"))
    R.test("knn_predict_np: döngü yok", lambda: max_loops(src, 0, func="knn_predict_np"))
    R.test("torch kullanılmıyor", lambda: no_imports(src, {"torch"}))
    rng = np.random.default_rng(1)
    A, B = rng.normal(size=(7, 5)), rng.normal(size=(11, 5))
    ref = ((A[:, None] - B[None]) ** 2).sum(-1)
    R.test("sqdist_one_loop doğru", lambda: close(need(ns, "sqdist_one_loop")(A, B), ref))
    R.test("sqdist_no_loop doğru ve ≥ 0", lambda: (close(need(ns, "sqdist_no_loop")(A, B), ref, rtol=1e-9, atol=1e-9),
                                                 ensure(np.all(need(ns, "sqdist_no_loop")(A, A) >= 0), "negatif mesafe var; np.maximum(D, 0) eksik")))

    def pred():
        f = need(ns, "knn_predict_np")
        ytr = rng.integers(0, 4, size=11)
        for k in (1, 3, 5, 11):
            got = np.asarray(f(ref, ytr, k)).tolist()
            ensure(got == ref_pred(ref, ytr, k), f"k={k}: {got} ≠ {ref_pred(ref, ytr, k)}")
        D = np.array([[1, 1, 2, 2], [3, 1, 1, 0]], dtype=float)
        y = np.array([1, 0, 0, 1])
        for k in (1, 2, 4):
            ensure(np.asarray(f(D, y, k)).tolist() == ref_pred(D, y, k), f"eşitlik durumu k={k}")
    R.test("knn_predict_np: rastgele + eşitlik kuralları", pred)
