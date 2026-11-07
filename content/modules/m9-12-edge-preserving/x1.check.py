import numpy as np
import cv2
from skimage import data
from labcheck import need, ensure


def check(ns, R):
    def exact():
        img = data.camera()[100:160, 300:370]
        got = need(ns, "ahe_sliding")(img, 9)
        P = cv2.copyMakeBorder(img, 4, 4, 4, 4, cv2.BORDER_REFLECT_101)
        want = np.zeros_like(img)
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                want[y, x] = np.rint(255 * (P[y:y + 9, x:x + 9] <= img[y, x]).mean())
        ensure(np.array_equal(got, want), "kayan pencere AHE")
    R.test("ahe_sliding", exact)

    def gain():
        rng = np.random.default_rng(0)
        img = np.full((128, 128), 120.0); img[:, 64:] = 180
        img = np.clip(img + rng.normal(0, 2, img.shape), 0, 255).astype(np.uint8)
        reg = (40, 90, 10, 50)
        he = cv2.equalizeHist(img); ahe = ns["ahe_sliding"](img, 31); cl = cv2.createCLAHE(2.0, (4, 4)).apply(img)
        g = {k: need(ns, "noise_gain")(img, v, reg) for k, v in (("HE", he), ("AHE", ahe), ("CLAHE", cl))}
        ensure(g["AHE"] > g["CLAHE"] and g["AHE"] > 5, f"kazançlar {g}")
        R.table("Düz bölge, giriş gürültüsü σ ≈ 2", ["yöntem", "gürültü kazancı"], [[k, f"{v:.1f}×"] for k, v in g.items()],
                note="AHE düz bölgenin dar histogramını tüm aralığa yayar: gürültü onlarca kat büyür. CLAHE eğimi sınırlar.")
    R.test("gürültü büyütme: AHE ≫ CLAHE", gain)
