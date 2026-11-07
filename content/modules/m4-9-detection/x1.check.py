import numpy as np
from labcheck import close, max_loops, need, ensure


def check(ns, R_):
    src = ns["__source__"]
    R_.test("döngüsüz", lambda: [max_loops(src, 0, func=f) for f in ("dense_to_conv", "fcn_apply", "sliding_dense")])

    def t():
        rng = np.random.default_rng(0)
        C, k, out = 10, 7, 20
        W, b = rng.normal(size=(out, C * k * k)), rng.normal(size=out)
        F7 = rng.normal(size=(C, 7, 7))
        Wc = need(ns, "dense_to_conv")(W, C, k)
        close(need(ns, "fcn_apply")(F7, Wc, b)[:, 0, 0], W @ F7.reshape(-1) + b, rtol=1e-10, what="7 × 7'de yoğun = konv. (1 × 1 çıkış)")
        F14 = rng.normal(size=(C, 14, 14))
        y = ns["fcn_apply"](F14, Wc, b)
        ensure(y.shape == (out, 8, 8), "14 × 14 → 8 × 8 ızgara")
        close(y, need(ns, "sliding_dense")(F14, W, b, k), rtol=1e-10, what="FCN = 64 konumda kayan yoğun katman")
        rows = []
        for s in (224, 320, 448, 640):
            g = s // 32 - 7 + 1
            rows.append([f"{s} × {s}", f"{s // 32} × {s // 32}", f"{g} × {g}", g * g, f"{(s / 224) ** 2:.1f}×", f"{g * g / (s / 224) ** 2:.1f}×"])
        R_.table("Toplam adımı 32 olan omurga: girdi boyutu → tahmin ızgarası", ["girdi", "darboğaz", "ızgara", "pencere", "FCN maliyeti (224'e göre)", "kayan pencereye göre kazanç"], rows,
                 note="Kayan pencere yaklaşımı ağı her pencere için baştan çalıştırır (maliyet ∝ pencere sayısı); FCN üst üste binen pencerelerin ortak hesabını paylaşır (maliyet ∝ görüntü alanı). 448'de 64 pencere 4× maliyete iner.")
    R_.test("FCN eşdeğerliği", t)
