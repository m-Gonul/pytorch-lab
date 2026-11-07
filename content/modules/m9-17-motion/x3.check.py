import numpy as np
from skimage import data
from labcheck import need, ensure


def check(ns, R):
    bg = data.camera()[200:296, 150:278].astype(np.float64) * 0.7
    obj = data.coins()[60:100, 60:100].astype(np.float64)
    fr = []
    for t in range(3):
        f = bg.copy(); f[28:68, 20 + 6 * t:60 + 6 * t] = obj
        fr.append(f)

    def t():
        P = lambda a: 10 * np.log10(255 ** 2 / ((a - fr[1]) ** 2).mean())
        rows = [[n, f"{P(need(ns, fn)(fr[0], fr[2])):.2f}"] for n, fn in [("tekrar", "fruc_repeat"), ("ortalama", "fruc_average"), ("simetrik blok eşleme", "fruc_block")]]
        v = [float(r[1]) for r in rows]
        ensure(v[2] > v[1] + 5 and v[2] > v[0] + 5, f"blok FRUC en iyi olmalı: {v}")
        R.table("Ara kare (nesne 12 px kayıyor)", ["yöntem", "PSNR (dB)"], rows,
                note="Simetrik arama ara karenin ızgarasında çalıştığı için delik ya da çakışma oluşmaz; gerçek videoda blok sınırlarında artefakt kalır (OBMC ya da yoğun akış çözümü).")
    R.test("FRUC karşılaştırması", t)
