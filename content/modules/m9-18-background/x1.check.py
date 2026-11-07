import numpy as np
from skimage import data
from labcheck import need, ensure


def check(ns, R):
    rng = np.random.default_rng(0)
    bg = data.camera()[::3, ::3][:120, :160].astype(np.float64)
    tex = data.coins()[60:90, 60:100].astype(np.float64)
    fr = []
    for t in range(3):
        f = bg.copy(); f[50:80, 20 + 15 * t:60 + 15 * t] = tex
        fr.append(np.clip(f + rng.normal(0, 5, f.shape), 0, 255).astype(np.uint8))

    def t():
        d2 = need(ns, "diff2")(fr[0], fr[1], 30)
        d3 = need(ns, "diff3")(fr[0], fr[1], fr[2], 30)
        obj = np.zeros(bg.shape, bool); obj[50:80, 35:75] = True
        ensure((d2 & ~obj).sum() > (d3 & ~obj).sum(), "üç kare farkı eski konumdaki 'kenar'ı atmalı")
        c = need(ns, "clean")(d3)
        b = need(ns, "boxes")(c)
        ensure(len(b) >= 1, "en az bir kutu")
        x, y, w, h = b[0]
        cx, cy = x + w / 2, y + h / 2
        ensure(35 <= cx <= 75 and 50 <= cy <= 80, f"kutu {b[0]} ortadaki karedeki nesnenin üzerinde olmalı")
        return f"iki kare farkı {int(d2.sum())} piksel, üç kare {int(d3.sum())}, temizlik + kutu: {b[0]}"
    R.test("fark, temizlik, kutu", t)
