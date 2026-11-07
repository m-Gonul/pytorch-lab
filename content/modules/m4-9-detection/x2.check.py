import numpy as np
from labcheck import close, max_loops, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("döngüsüz", lambda: (max_loops(src, 0, func="fpn_level"), max_loops(src, 0, func="upsample2")))

    def t_lvl():
        sizes = [16, 32, 64, 112, 160, 224, 320, 448, 900]
        lv = need(ns, "fpn_level")(np.array(sizes), np.array(sizes))
        ensure(list(lv) == [2, 2, 2, 3, 3, 4, 4, 5, 5], f"seviyeler {list(lv)}")
        ensure(int(ns["fpn_level"](100, 50)) == 2, "100 × 50 → P2")
        R_.table("RoI boyutu → FPN seviyesi (k₀ = 4)", ["RoI kenarı", "seviye", "adım"], [[s, f"P{l}", 2 ** l] for s, l in zip(sizes, lv)],
                 note="RoI kenarı yarıya inince bir alt (daha ince) seviyeye geçilir: her seviyede nesne yaklaşık aynı sayıda öznitelik hücresi kaplar — görüntü piramidinin ölçek değişmezliği, ağın içinde ve tek geçişte.")
    R_.test("seviye ataması", t_lvl)

    def t_td():
        rng = np.random.default_rng(0)
        ch = [8, 16, 32, 64]
        H = [32, 16, 8, 4]
        d = 6
        feats = [rng.normal(size=(c, h, h)) for c, h in zip(ch, H)]
        lat = [rng.normal(size=(d, c)) / np.sqrt(c) for c in ch]
        sm = [rng.normal(size=(d, d, 3, 3)) / 6 for _ in ch]
        close(need(ns, "upsample2")(np.arange(4.).reshape(1, 2, 2))[0], np.array([[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3.]]), what="en yakın komşu")
        P = need(ns, "top_down")(feats, lat, sm)
        ensure([p.shape for p in P] == [(d, h, h) for h in H], "P2–P5 şekilleri")
        f2 = [f.copy() for f in feats]
        f2[3][:, 0, 0] += 1.0
        P2 = ns["top_down"](f2, lat, sm)
        ch_ = [np.abs(a - b).max(0) > 1e-12 for a, b in zip(P, P2)]
        ensure(ch_[0][:6, :6].any() and not ch_[0][10:, 10:].any(), "C5'teki tek hücre P2'de yalnız karşılık gelen 8 × 8 bölgeyi (+ 3×3 kenar) etkiler")
        f3 = [f.copy() for f in feats]
        f3[0][:, 5, 5] += 1.0
        P3 = ns["top_down"](f3, lat, sm)
        ensure(all(np.allclose(a, b) for a, b in zip(P[1:], P3[1:])), "C2 değişikliği üst seviyelere sızmaz (yalnız yukarıdan aşağı akış)")
    R_.test("yukarıdan-aşağı yol", t_td)
