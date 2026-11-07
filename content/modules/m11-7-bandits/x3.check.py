import random
from labcheck import need, ensure


def check(ns, R_):
    def t():
        rng = random.Random(5)
        ctr = [0.01 + 0.04 * rng.random() for _ in range(20)]
        T = 20000
        best = max(ctr) * T
        u = sum(need(ns, "ads")("ucb", ctr, T, s) for s in range(3)) / 3
        th = sum(ns["ads"]("ts", ctr, T, s) for s in range(3)) / 3
        rnd = sum(ctr) / 20 * T
        ensure(th > u, "küçük tıklama oranlarında Thompson UCB1'den iyi")
        p, loss = need(ns, "ab_test")(30, 1000, 45, 1000)
        ensure(0.93 < p < 0.98 and loss < 0.001, "A/B: P(B > A) ≈ 0.96")
        p2, _ = ns["ab_test"](30, 1000, 30, 1000)
        ensure(abs(p2 - 0.5) < 0.01, "eşit veride ≈ 0.5")
        R_.table("20 reklam, tıklama oranı %1–%5, 20 000 gösterim (3 tohum)", ["strateji", "tıklama"],
                 [["rastgele (beklenen)", f"{rnd:.0f}"], ["UCB1", f"{u:.0f}"], ["Thompson", f"{th:.0f}"], ["en iyi reklam hep (oracle)", f"{best:.0f}"]],
                 note=f"UCB1'in Hoeffding terimi [0,1] aralığını varsaydığı için %3'lük oranlarda aşırı keşfeder. A/B testi (30/1000 vs 45/1000): P(B > A) = {p:.3f}, B'yi seçmenin beklenen kaybı {loss:.5f}.")
    R_.test("reklam ve A/B", t)
