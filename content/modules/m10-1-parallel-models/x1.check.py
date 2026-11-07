import math
from labcheck import need, ensure


def check(ns, R_):
    def t():
        rows = []
        for n in (8, 64, 1024):
            dur, preds = need(ns, "mergesort_dag")(n)
            W = sum(dur.values())
            memo = {}

            def ef(v):
                if v not in memo:
                    memo[v] = dur[v] + max((ef(u) for u in preds[v]), default=0)
                return memo[v]

            S = max(ef(v) for v in preds)
            ensure(W == n + n * int(math.log2(n)), "iş = n + n log n")
            ensure(S == 1 + 2 * n - 2, "yol = 1 + (n + n/2 + … + 2)")
            mc = need(ns, "max_concurrency")(dur, preds)
            ensure(mc == n, "en çok eşzamanlılık: n yaprak")
            rows.append([n, W, S, f"{W / S:.1f}", mc])
        R_.table("Üstten aşağı merge sort (birleştirme sıralı)", ["n", "iş W", "yol S∞", "paralellik", "en çok eşzamanlı görev"], rows,
                 note="Birleştirme adımı sıralı kaldıkça yol O(n): paralellik yalnız O(log n). Paralel birleştirme (ikili aramayla bölme) yolu O(log² n)'ye indirir.")
    R_.test("merge sort grafiği", t)
