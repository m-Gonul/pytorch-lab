from labcheck import need, ensure


def check(ns, R_):
    def t():
        rows = []
        for lam, mu, c in [(0.8, 1.0, 1), (1.8, 1.0, 2), (3.5, 1.0, 4), (7.0, 1.0, 8)]:
            sim = sum(need(ns, "simulate")(lam, mu, c, 30000, s) for s in range(3)) / 3
            th = need(ns, "erlang_c_wait")(lam, mu, c)
            ensure(abs(sim - th) / th < 0.15, f"simülasyon ≈ Erlang C (c = {c}: {sim:.2f} vs {th:.2f})")
            rows.append([c, lam, f"{lam / (c * mu):.2f}", f"{th:.3f}", f"{sim:.3f}"])
        R_.table("M/M/c kuyruğu, ortalama bekleme", ["gişe c", "λ", "yük ρ", "Erlang C", "simülasyon"], rows,
                 note="Aynı yükte (ρ ≈ 0.9) gişe sayısı arttıkça bekleme düşer: havuzlanmış kaynaklar dalgalanmaları emer — iş havuzlu (dinamik) zamanlamanın kuyruk teorisi karşılığı. Olay kuyruğu düzensiz, önceden bilinmeyen iletişimi modellemenin klasik yoludur.")
    R_.test("ayrık olay simülasyonu", t)
