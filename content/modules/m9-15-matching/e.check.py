import time
import numpy as np
import cv2
import torch
from skimage import data
from labcheck import need, ensure, bench, fmt_time


def views():
    big = data.astronaut()
    H1 = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1]])
    A = np.array([[0.97, -0.06, 0], [0.05, 0.98, 0], [1.2e-4, -0.8e-4, 1]])
    T = np.array([[1, 0, -220], [0, 1, -40], [0, 0, 1]])
    H2 = A @ T
    v1 = cv2.warpPerspective(big, H1, (320, 360))
    v2 = cv2.warpPerspective(big, H2, (320, 360))
    return big, v1, v2, H1 @ np.linalg.inv(H2)


def check(ns, R):
    big, v1, v2, Ht = views()

    def rs():
        rng = np.random.default_rng(0)
        src = rng.uniform(0, 500, (300, 2))
        dst = cv2.perspectiveTransform(src[None], Ht)[0] + rng.normal(0, 0.5, (300, 2))
        out = rng.random(300) < 0.5
        dst[out] = rng.uniform(0, 500, (out.sum(), 2))
        H, m = need(ns, "ransac_t")(src, dst, 3.0, 1000)
        c = np.array([[[0, 0], [320, 0], [320, 360], [0, 360]]], np.float64)
        e = np.abs(cv2.perspectiveTransform(c, H.numpy()) - cv2.perspectiveTransform(c, Ht)).max()
        ensure(e < 3 and (m.numpy() == ~out).mean() > 0.95, f"köşe hatası {e:.2f}")
        t1 = bench(lambda: ns["ransac_t"](src, dst, 3.0, 1000), repeat=3)
        t2 = bench(lambda: cv2.findHomography(src, dst, cv2.RANSAC, 3.0), repeat=3)
        R.table("%50 aykırı, 300 eş", ["yöntem", "süre"], [["ransac_t (1000 hipotez tek SVD'de)", fmt_time(t1)], ["cv2.findHomography RANSAC (uyarlamalı iterasyon)", fmt_time(t2)]])
    R.test("ransac_t", rs)

    def st():
        pano, H = need(ns, "stitch")(v1, v2)
        c = np.array([[[0, 0], [319, 0], [319, 359], [0, 359]]], np.float64)
        e = np.abs(cv2.perspectiveTransform(c, H.numpy()) - cv2.perspectiveTransform(c, Ht)).max()
        ensure(e < 2.0, f"köşe hatası {e:.2f} piksel")
        ensure(pano.dtype == np.uint8 and pano.shape[1] > 450, f"panorama boyutu {pano.shape}")
        ov = big[60:300, 230:300].astype(float)
        P = pano.astype(float)
        best = max(10 * np.log10(255 ** 2 / ((P[60 + dy:300 + dy, 230 + dx:300 + dx] - ov) ** 2).mean()) for dy in range(-2, 3) for dx in range(-2, 3))
        ensure(best > 25, f"örtüşme bölgesi PSNR {best:.1f}")
        return f"panorama {pano.shape[1]} × {pano.shape[0]}, homografi köşe hatası {e:.2f} px, örtüşmede PSNR {best:.1f} dB"
    R.test("stitch (alıştırma 6)", st)
