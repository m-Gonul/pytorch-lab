import numpy as np
from labcheck import close, max_loops, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("döngüsüz", lambda: [max_loops(src, 0, func=f) for f in ("window_partition", "window_reverse", "shifted_window_mask", "attn_cost")])

    def t():
        rng = np.random.default_rng(0)
        x = rng.normal(size=(8, 12, 3))
        w = need(ns, "window_partition")(x, 4)
        ensure(w.shape == (6, 16, 3), "6 pencere × 16 token")
        close(w[1], x[0:4, 4:8].reshape(16, 3), what="pencere 1 = satırlar 0–3, sütunlar 4–7")
        close(need(ns, "window_reverse")(w, 4, 8, 12), x, what="ters")
        m = need(ns, "shifted_window_mask")(8, 8, 4, 2)
        ensure(m.shape == (4, 16, 16), "maske şekli")
        ensure(m[0].all(), "sol üst pencere tek bölge")
        ensure(m[3].sum() == 4 * 16 + 4 * 16 + 0 or m[3].sum() == (4 + 4 + 4 + 4) * 4, "sağ alt pencere 4 bölgeye bölünür (her biri 2 × 2 = 4 token)")
        ensure(m[1].sum() == 8 * 8 + 8 * 8, "sağ üst pencere 2 bölge (8'er token)")
        rows = []
        for H, C, M in ((56, 96, 7), (112, 96, 7), (224, 96, 7)):
            g, wn = need(ns, "attn_cost")(H, H, C, M)
            rows.append([f"{H} × {H}", f"{g / 1e9:.1f}", f"{wn / 1e9:.2f}", f"{g / wn:.0f}×"])
        R_.table("Swin'in ilk aşaması (C = 96, M = 7): küresel vs pencere MSA (GMAC)", ["öznitelik haritası", "küresel", "pencere", "oran"], rows,
                 note="Küresel MSA token sayısının karesiyle, pencere MSA doğrusal büyür (Swin Denk. 1–2). Pencereler arası bilgi akışı için her ikinci blokta pencereler ⌊M/2⌋ kaydırılır; kaydırılmış görüntüde bir pencere, aslında bitişik olmayan bölgeleri içerebilir — maske yalnız aynı bölgedeki token'ların birbirine bakmasına izin verir, böylece pencere sayısı artmadan (döngüsel kaydırma) hesap korunur.")
    R_.test("pencere attention'ı", t)
