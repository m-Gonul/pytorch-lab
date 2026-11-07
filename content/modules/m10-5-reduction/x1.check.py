from labcheck import need, ensure


def check(ns, R_):
    def t():
        rows = []
        for n in (10 ** 3, 10 ** 5, 10 ** 6):
            e = need(ns, "errors")(n, 1)
            rows.append([n] + [f"{v:.1e}" for v in e])
        last = [float(v) for v in rows[-1][1:]]
        ensure(last[2] < 1e-6 and last[2] < last[0] / 100, "Kahan çok daha doğru")
        ensure(float(need(ns, "kahan_sum")([1.0] + [1e-8] * 1000)) > 1.0, "Kahan küçük terimleri kaybetmez")
        R_.table("float32 toplamın göreli hatası", ["n", "sıralı", "ikili (ağaç)", "Kahan", "np.sum"], rows,
                 note="np.sum zaten 8'li bloklarla ikili toplama yapar. Kahan her adımda kaybolan alçak bitleri c'de taşır; bedeli 4 kat FP işlem ve derleyicinin 'akıllı' cebirsel sadeleştirmesine kapalı olmak (-ffast-math Kahan'ı bozar).")
    R_.test("toplam hassasiyeti", t)
