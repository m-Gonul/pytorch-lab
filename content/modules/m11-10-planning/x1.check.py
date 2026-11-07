from labcheck import need, ensure


def check(ns, R_):
    def t():
        res = {}
        for plus in (False, True):
            R = [need(ns, "run")(plus, 6000, 3000, 1e-3, s) for s in range(5)]
            res[plus] = [sum(r[k] for r in R) / 5 for k in (999, 2999, 3999, 4999, 5999)]
        slope = {p: (v[-1] - v[1]) / 3000 for p, v in res.items()}
        ensure(slope[True] > 1.25 * slope[False], f"Dyna-Q+ kısayolu kullanmalı (eğimler {slope})")
        R_.table("Kısayol labirenti, kümülatif ödül (5 deney); 3000. adımda kısayol açılır", ["yöntem", "1000", "3000", "4000", "5000", "6000", "sonrası adım/bölüm"],
                 [["Dyna-Q" if not p else "Dyna-Q+"] + [f"{x:.0f}" for x in v] + [f"{1 / slope[p]:.1f}"] for p, v in res.items()],
                 note="Dyna-Q eski (hâlâ çalışan) yolu kullanmaya devam eder; Dyna-Q+ uzun süredir denenmeyen geçişleri planlamada cazip kılar, kısayolu keşfeder (Sutton Şekil 8.5).")
    R_.test("Dyna-Q+", t)
