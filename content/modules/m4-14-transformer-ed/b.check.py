import math
from labcheck import close, no_imports, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("numpy kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))

    def t():
        P = need(ns, "positional_encoding")(64, 512)
        close(P[1][0], math.sin(1), what="P[1,0]")
        close(P[2][1], math.cos(2), what="P[2,1]")
        close(P[22][100], P[60][100], atol=0.05, what="38 konum arayla boyut 100 neredeyse aynı (Géron)")
        close(P[22][101], P[60][101], atol=0.05, what="boyut 101")
        ensure(len(set(tuple(round(v, 6) for v in row) for row in P)) == 64, "her konum benzersiz")
        close(need(ns, "noam_lr")(4000, 512, 4000), 512 ** -0.5 * 4000 ** -0.5, what="Noam tepe")
        ensure(ns["noam_lr"](2000, 512, 4000) < ns["noam_lr"](4000, 512, 4000) > ns["noam_lr"](16000, 512, 4000), "ısınma sonra azalma")
        close(ns["noam_lr"](16000, 512, 4000) / ns["noam_lr"](4000, 512, 4000), 0.5, what="4× adımda yarıya (1/√s)")
        close(need(ns, "smooth_targets")(1, 5, 0.1), [0.02, 0.92, 0.02, 0.02, 0.02], what="hedef")
        q = ns["smooth_targets"](1, 5, 0.1)
        close(need(ns, "smoothed_ce")([math.log(x) for x in q], 1, 0.1), -sum(x * math.log(x) for x in q), what="en iyi tahminde CE = H(q)")
        p = need(ns, "transformer_params")(512, 8, 2048, 6, 37000)
        ensure(p == {"enc_layer": 3152384, "dec_layer": 4204032, "total": 63082496}, f"base {p}")
        ensure(need(ns, "causal_allowed")(4) == 10, "nedensel")
        rows = []
        for name, (d, h, ff, N) in (("base", (512, 8, 2048, 6)), ("big", (1024, 16, 4096, 6)), ("d aşamasındaki", (64, 4, 128, 2))):
            pp = ns["transformer_params"](d, h, ff, N, 37000 if name != "d aşamasındaki" else 24)
            rows.append([name, d, N, f"{pp['total'] / 1e6:.2f} M", f"{37000 * d / pp['total']:.0%}" if name != "d aşamasındaki" else "—"])
        R_.table("Parametre sayımları", ["model", "d", "N", "toplam", "embedding payı"], rows,
                 note="Base modelde parametrelerin ~%30'u paylaşılan embedding'dedir; paylaşım olmasaydı (kaynak, hedef, çıktı ayrı) 37M daha eklenirdi. big (213M makalede) d'yi iki katına çıkarır: katman parametreleri ~4×.")
    R_.test("formüller", t)
