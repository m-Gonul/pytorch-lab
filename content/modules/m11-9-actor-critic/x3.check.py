import math
from labcheck import need, ensure


def check(ns, R_):
    def t():
        tg = need(ns, "true_grad")()
        rows = []
        var = {}
        for kind, name in [("G", "taban çizgisiz"), ("b", "sabit b = J(p)"), ("V", "durum taban çizgisi V(s)")]:
            xs = need(ns, "grad_samples")(20000, kind, 1)
            m = sum(xs) / len(xs)
            v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
            se = math.sqrt(v / len(xs))
            ensure(abs(m - tg) < 4 * se, f"{name}: yansız olmalı ({m:.2f} vs {tg:.2f})")
            var[kind] = v
            rows.append([name, f"{m:.3f}", f"{se:.3f}", f"{v:.1f}"])
        ensure(var["V"] < var["G"], "durum taban çizgisi varyansı düşürmeli")
        R_.table(f"∂J/∂θ₀ kestiricileri, p = 0.3, 20 000 bölüm (gerçek: {tg:.3f})", ["kestirici", "ortalama", "std. hata", "varyans"], rows,
                 note="Üçü de yansız; varyans farkı aynı doğruluk için gereken bölüm sayısının oranıdır.")
    R_.test("varyans", t)
