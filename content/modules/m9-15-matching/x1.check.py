import numpy as np
import cv2
from skimage import data
from labcheck import need, ensure


def check(ns, R):
    img = data.camera().astype(np.float32)
    locs = [(250, 200), (120, 380), (400, 60), (330, 300)]
    names = {cv2.TM_SQDIFF: "SQDIFF", cv2.TM_SQDIFF_NORMED: "SQDIFF_NORMED", cv2.TM_CCORR: "CCORR", cv2.TM_CCORR_NORMED: "CCORR_NORMED", cv2.TM_CCOEFF: "CCOEFF", cv2.TM_CCOEFF_NORMED: "CCOEFF_NORMED"}

    def modes():
        rows = []
        for m, n in names.items():
            ok = 0
            for x, y in locs:
                t = img[y:y + 40, x:x + 50].copy()
                for g, o in [(1.0, 0), (0.6, 60), (1.3, -30)]:
                    search = img.copy(); search[y:y + 40, x:x + 50] = np.clip(t * g + o, 0, 255)
                    ok += need(ns, "best_match")(search, t, m) == (x, y)
            rows.append([n, f"{ok}/12"])
        d = dict((r[0], int(r[1].split("/")[0])) for r in rows)
        ensure(d["CCOEFF_NORMED"] == 12, "korelasyon katsayısı her durumda bulmalı")
        R.table("Hedef bölgeye kazanç/ofset uygulanınca doğru bulma", ["yöntem", "doğru"], rows, note="Ham CCORR parlak bölgelere, SQDIFF ofsete takılır; yalnız ortalaması çıkarılmış ve normalize edilmiş ölçü hepsini bulur.")
    R.test("eşleme yöntemleri", modes)

    def ms():
        t = img[200:248, 260:320].copy()
        small = cv2.resize(img, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)
        x, y, s, sc = need(ns, "multiscale")(small, t, np.arange(0.5, 1.01, 0.05))
        ensure(abs(s - 0.7) < 0.051 and abs(x - 260 * 0.7) <= 3 and abs(y - 200 * 0.7) <= 3, f"bulunan ({x}, {y}), ölçek {s:.2f}")
        return f"ölçek {s:.2f}, skor {sc:.3f}"
    R.test("çok ölçekli eşleme", ms)
