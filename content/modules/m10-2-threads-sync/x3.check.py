from labcheck import need, ensure


def check(ns, R_):
    def t():
        rows = []
        res = {}
        for pol in ("reader", "writer"):
            ww, wr, dw = need(ns, "simulate")(pol, 0.6, 0.05, 20000, 1)
            res[pol] = (ww, wr, dw)
            rows.append([pol, f"{ww:.1f}", f"{wr:.1f}", dw])
        ensure(res["reader"][0] > 5 * res["writer"][0], "okuyucu önceliğinde yazıcılar aç kalmalı")
        ensure(res["writer"][1] > res["reader"][1], "yazıcı önceliğinde okuyucular daha çok bekler")
        R_.table("Okuyucu–yazıcı kilidi: p_okuyucu = 0.6, p_yazıcı = 0.05 / tik, 20 000 tik", ["öncelik", "ort. yazıcı beklemesi", "ort. okuyucu beklemesi", "biten yazıcı"], rows,
                 note="Sürekli okuyucu akışı varken okuyucu öncelikli kilit yazıcıya hiç boşluk bırakmaz (açlık). Adil kilitler gelen sırasına göre (FIFO) davranarak ikisinin ortasında durur; C++17 std::shared_mutex'in politikası tanımsızdır.")
    R_.test("okuyucu–yazıcı", t)
