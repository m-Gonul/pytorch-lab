from labcheck import close, need, ensure


def check(ns, R_):
    def t():
        s = need(ns, "semi_gradient")(1.0, 0.1, 0.9, 50)
        close(s[0], 1 + 0.1 * 0.8, what="ilk adım: (1 + α(2γ − 1))w")
        close(s[-1], (1 + 0.1 * 0.8) ** 50, rtol=1e-9, what="geometrik büyüme")
        r = need(ns, "residual_gradient")(1.0, 0.1, 0.9, 50)
        close(r[-1], (1 - 0.1 * 0.64) ** 50, rtol=1e-9, what="kalıntı gradyan küçülür")
        o = need(ns, "on_policy")(1.0, 0.1, 0.9, 50)
        close(o[0], (1 + 0.08) * (1 - 0.4), what="politika üstü bölüm: (1 + α(2γ − 1))(1 − 4α)")
        rows = []
        for g in (0.4, 0.5, 0.6, 0.9, 0.99):
            rows.append([g, f"{ns['semi_gradient'](1.0, 0.1, g, 100)[-1]:.3g}", f"{ns['residual_gradient'](1.0, 0.1, g, 100)[-1]:.3g}", f"{ns['on_policy'](1.0, 0.1, g, 100)[-1]:.3g}"])
        R_.table("w (başlangıç 1, α = 0.1), 100 güncelleme / bölüm", ["γ", "yarı-gradyan, politika dışı", "kalıntı gradyan", "yarı-gradyan, politika üstü"], rows,
                 note="γ > 0.5'te yalnız s₁→s₂ geçişini güncellemek ıraksar: hedef 2γw, kestirim w'den hızlı büyür. s₂'nin kendi güncellemesi de yapılırsa (politika üstü dağılım) w sıfıra gider. Kalıntı gradyan her zaman kararlıdır ama stokastik ortamda yanlı ve yavaştır.")
    R_.test("w → 2w", t)
