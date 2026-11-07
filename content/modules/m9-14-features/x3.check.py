import numpy as np
import cv2
from skimage import data
from labcheck import need, ensure


def check(ns, R):
    def mr():
        rng = np.random.default_rng(0)
        a = rng.normal(size=(50, 16)); b = np.concatenate([a + 0.01 * rng.normal(size=a.shape), rng.normal(size=(30, 16))])
        m = need(ns, "match_ratio")(a, b)
        ensure(len(m) == 50 and all(i == j for i, j in m), "gürültülü kopyalar eşleşmeli")
        ab = rng.integers(0, 256, (20, 32), dtype=np.uint8)
        mb = ns["match_ratio"](ab, ab.copy())
        ensure(len(mb) == 20 and all(i == j for i, j in mb), "ikili betimleyici (Hamming)")
    R.test("match_ratio", mr)

    def ev():
        img = data.camera()
        rows = []
        for ang, sc in [(0, 1.0), (30, 1.0), (45, 0.7), (90, 1.3)]:
            M = cv2.getRotationMatrix2D((256, 256), ang, sc)
            s = need(ns, "evaluate")(img, M, "sift"); o = ns["evaluate"](img, M, "orb")
            rows.append([f"{ang}°, ×{sc}", f"{s[1]}/{s[0]}", f"{o[1]}/{o[0]}"])
            ensure(s[0] > 0 and s[1] / s[0] > 0.7, f"SIFT doğru oranı düşük ({ang}°, ×{sc})")
        R.table("Doğru / toplam eşleşme (oran testi 0.8)", ["dönüşüm", "SIFT", "ORB"], rows,
                note="Alıştırma 6'nın panoraması bu eşleşmelerden RANSAC ile homografi kestirir (M9.15).")
    R.test("SIFT ve ORB eşleştirme", ev)
