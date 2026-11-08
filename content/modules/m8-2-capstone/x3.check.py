import numpy as np
from labcheck import close, max_loops, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("döngüsüz", lambda: [max_loops(src, 0, func=f) for f in ("paired_bootstrap", "unpaired_bootstrap")])

    def t():
        rng = np.random.default_rng(0)
        d, lo, hi = need(ns, "paired_bootstrap")(np.ones(10) * 2, np.ones(10), 200, rng)
        close((d, lo, hi), (1.0, 1.0, 1.0), what="sabit fark")
        cover, widths_p, widths_u, sig_p, sig_u = 0, [], [], 0, 0
        for k in range(200):
            hard = rng.random(300)
            a = np.clip(0.9 - 0.6 * hard + 0.05 * rng.normal(size=300), 0, 1)
            b = np.clip(a - 0.01 + 0.03 * rng.normal(size=300), 0, 1)
            dp, lp, hp = ns["paired_bootstrap"](a, b, 400, rng)
            du, lu, hu = need(ns, "unpaired_bootstrap")(a, b, 400, rng)
            cover += lp <= 0.01 <= hp
            widths_p.append(hp - lp)
            widths_u.append(hu - lu)
            sig_p += lp > 0
            sig_u += lu > 0
        ensure(0.88 <= cover / 200 <= 0.99, f"%95 aralığın kapsama oranı {cover / 200:.2f}")
        ensure(np.mean(widths_p) < 0.3 * np.mean(widths_u), "eşleştirme aralığı daraltır")
        R_.table("200 sentetik 'deney': örnek başına skorlar, gerçek ortalama fark 0.01 (model A, B'den biraz iyi)", ["yöntem", "ort. %95 aralık genişliği", "farkı anlamlı bulma oranı"], [["eşleştirilmiş bootstrap", f"{np.mean(widths_p):.4f}", f"{sig_p / 200:.2f}"], ["eşleştirilmemiş bootstrap", f"{np.mean(widths_u):.4f}", f"{sig_u / 200:.2f}"]],
                 note=f"Örnek zorluğu iki modeli birlikte etkiler (zor görüntüde ikisi de kötü): bu ortak varyans eşleştirilmemiş karşılaştırmada gürültü olarak kalır, küçük ama gerçek bir farkı gizler. Aynı test örnekleri üzerinde farkları yeniden örneklemek bu varyansı yok eder. %95 aralığın kapsama oranı {cover / 200:.2f}. Raporlarken tohum varyansını da (birkaç eğitim koşusu) ekle — bootstrap yalnız test kümesi belirsizliğini ölçer.")
    R_.test("eşleştirilmiş bootstrap", t)
