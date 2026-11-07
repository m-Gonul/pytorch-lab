import random
from labcheck import close, no_imports, need, ensure


def ref_pred(D, y, k):
    out = []
    for row in D:
        order = sorted(range(len(row)), key=lambda j: (row[j], j))[:k]
        cnt = {}
        for j in order:
            cnt[y[j]] = cnt.get(y[j], 0) + 1
        out.append(min(cnt, key=lambda c: (-cnt[c], c)))
    return out


def check(ns, R):
    src = ns["__source__"]
    R.test("numpy/torch kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))
    Xtr, ytr = [[0, 0], [1, 3], [3, 1], [4, 4]], [0, 1, 0, 1]
    R.test("elle örnek: [[5, 1, 5, 13]]", lambda: close(need(ns, "sqdist_loops")([[1, 2]], Xtr), [[5, 1, 5, 13]]))

    def hand_pred():
        D = [[5, 1, 5, 13]]
        ensure(list(need(ns, "knn_predict")(D, ytr, 1)) == [1], "k=1 için [1] bekleniyordu")
        ensure(list(need(ns, "knn_predict")(D, ytr, 3)) == [0], "k=3 için [0] bekleniyordu")
    R.test("elle örnek: k=1 → 1, k=3 → 0", hand_pred)

    def rnd():
        rng = random.Random(0)
        A = [[rng.uniform(-2, 2) for _ in range(3)] for _ in range(6)]
        B = [[rng.uniform(-2, 2) for _ in range(3)] for _ in range(9)]
        close(need(ns, "sqdist_loops")(A, B), [[sum((a - b) ** 2 for a, b in zip(x, z)) for z in B] for x in A], what="6×9 mesafe")
    R.test("rastgele 6×9 mesafe matrisi", rnd)

    def ties():
        D = [[1, 1, 2, 2], [3, 1, 1, 0]]
        y = [1, 0, 0, 1]
        for k in (1, 2, 4):
            got = list(need(ns, "knn_predict")(D, y, k))
            ensure(got == ref_pred(D, y, k), f"k={k}: {got} bulundu, {ref_pred(D, y, k)} bekleniyordu (eşitlik kuralları)")
    R.test("eşitlik kuralları (mesafe ve oy)", ties)
