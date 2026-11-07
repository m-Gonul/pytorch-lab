import random
from labcheck import close, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("numpy/torch kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))
    rng = random.Random(0)

    def t(N, c):
        def f():
            vecs = [[rng.randint(-9, 9) for _ in range(N * c)] for _ in range(N)]
            total = [sum(v[j] for v in vecs) for j in range(N * c)]
            res, sent = need(ns, "ring_allreduce")([list(v) for v in vecs])
            for i in range(N):
                ensure(list(res[i]) == total, f"işçi {i} toplamı yanlış")
            ensure(list(sent) == [2 * (N - 1) * c] * N, f"gönderilen {sent}, beklenen her işçi {2 * (N - 1) * c}")
            return f"N = {N}: işçi başına {2 * (N - 1) * c} / {N * c} eleman = {2 * (N - 1) / N:.2f}×"
        return f
    R.test("N = 2", t(2, 3))
    R.test("N = 4", t(4, 2))
    R.test("N = 8", t(8, 5))

    def orig():
        vecs = [[1, 2], [3, 4]]
        need(ns, "ring_allreduce")(vecs)
    R.test("girdi değişmese de olur (kopya)", orig)
