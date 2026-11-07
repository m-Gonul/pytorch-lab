from labcheck import need, ensure


def check(ns, R_):
    def t():
        a = need(ns, "nonstationary")(1000, 5000, None, 1)
        b = ns["nonstationary"](1000, 5000, 0.1, 1)
        ensure(b[-500:].mean() > a[-500:].mean() + 0.1, "sabit α durağan olmayan durumda daha iyi")
        R_.table("Durağan olmayan haydut, optimal eylem oranı (1000 deney)", ["adım", "örnek ortalaması", "α = 0.1"],
                 [[k, f"{100 * a[k - 100:k].mean():.0f}%", f"{100 * b[k - 100:k].mean():.0f}%"] for k in (500, 1000, 2500, 5000)],
                 note="1/n adımları eski ödülleri eşit ağırlıkla tutar; sabit α üstel ağırlıklı yakın-geçmiş ortalamasıdır ve değişen q*'ı izler — ama Robbins–Monro'nun Σα² < ∞ koşulunu bilerek bozar.")
    R_.test("durağan olmayan haydut", t)
