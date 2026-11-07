from labcheck import close, need, ensure


def check(ns, R_):
    def t():
        st = [2, 3, 5, 2, 3]
        close(need(ns, "simulate")(st, [1] * 5, 100), 510, what="çoğaltmasız = el hesabı")
        rows = []
        for rep in ([1, 1, 1, 1, 1], [1, 1, 2, 1, 1], [1, 2, 2, 1, 2], [1, 2, 3, 1, 2]):
            T = ns["simulate"](st, rep, 1000)
            thr = need(ns, "throughput")(st, rep)
            ensure(abs(1000 / T - thr) / thr < 0.02, "uzun akışta simülasyon = darboğaz formülü")
            rows.append([str(rep), sum(rep), f"{T:.0f}", f"{thr:.3f}"])
        R_.table("Boru hattı (2, 3, 5, 2, 3), 1000 öğe", ["çoğaltma", "işçi", "toplam süre", "verim (öğe/zaman)"], rows,
                 note="Verim en yavaş (çoğaltılmış) aşamaya bağlı: 5'lik aşamayı ikiye bölmek verimi 0.2'den 0.33'e çıkarır, artık darboğaz 3'lük aşamalardır.")
    R_.test("aşama çoğaltma", t)
