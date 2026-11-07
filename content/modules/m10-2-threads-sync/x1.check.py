from labcheck import need, ensure


def check(ns, R_):
    def t():
        rows = []
        for n in (2, 3, 4, 5, 6):
            d0, s0 = need(ns, "philosophers")(n, False)
            d1, s1 = ns["philosophers"](n, True)
            ensure(d0 and not d1, f"n = {n}: saf sırada kilitlenme var, sıralı almada yok")
            rows.append([n, s0, "VAR", s1, "yok"])
        R_.table("Yemek yiyen filozoflar (tek tur), tüm iç içe geçmeler", ["n", "durum (sol→sağ)", "kilitlenme", "durum (küçük çatal önce)", "kilitlenme"], rows,
                 note="Herkes sol çatalını alınca döngüsel bekleme oluşur. Küresel bir kaynak sırası döngüyü imkânsız kılar: en büyük numaralı çatalı bekleyen hep ilerleyebilir.")
    R_.test("filozoflar", t)
