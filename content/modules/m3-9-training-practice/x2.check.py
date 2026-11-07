from labcheck import need, ensure


def check(ns, R_):
    def t():
        sizes = [15, 30, 60, 120, 500, 2000]
        rows = []
        res = {}
        for d in (1, 5, 15):
            tr, va = need(ns, "learning_curve")(sizes, d, 1e-8, 1)
            res[d] = (tr, va)
            rows.append([d] + [f"{a:.3f}/{b:.3f}" for a, b in zip(tr, va)])
        ensure(need(ns, "diagnose")(res[1][0][-1], res[1][1][-1], 0.04) == "yüksek yanlılık", "derece 1: yanlılık")
        ensure(ns["diagnose"](res[15][0][1], res[15][1][1], 0.04) == "yüksek varyans", "derece 15, az veri: varyans")
        ensure(res[15][1][-1] < res[15][1][1], "varyans daha çok veriyle azalır")
        R_.table("Eğitim / doğrulama MSE (gürültü varyansı 0.04)", ["derece"] + [f"n = {n}" for n in sizes], rows,
                 note="Yanlılık: iki eğri yüksek ve birbirine yakın — daha çok veri yardım etmez, daha büyük model gerek. Varyans: büyük boşluk — veri ya da düzenleme yardım eder.")
    R_.test("öğrenme eğrileri", t)
