import numpy as np
import cv2
from skimage import data
from labcheck import close, no_loops, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("döngü yok", lambda: no_loops(src))
    clean = data.camera()
    rng = np.random.default_rng(0)
    noisy_sp = clean.copy(); u = rng.random(clean.shape); noisy_sp[u < 0.05] = 0; noisy_sp[(u >= 0.05) & (u < 0.1)] = 255

    def med():
        for k in (3, 5, 7):
            ensure(np.array_equal(need(ns, "median_np")(noisy_sp, k), cv2.medianBlur(noisy_sp, k)), f"{k}×{k}")
    R.test("median_np = cv2.medianBlur (3, 5, 7)", med)

    def ps():
        out = cv2.GaussianBlur(noisy_sp, (5, 5), 0)
        close(need(ns, "psnr")(clean, out), cv2.PSNR(clean, out), atol=1e-6)
        close(ns["psnr"](clean, clean), cv2.PSNR(clean, clean), what="özdeş görüntüler")
    R.test("psnr = cv2.PSNR", ps)

    def am():
        sig = np.sqrt(0.001225)
        noisy = np.clip(np.rint((clean / 255 + rng.normal(0, sig, clean.shape)) * 255), 0, 255).astype(np.uint8)
        got = need(ns, "adaptive_mean")(noisy, 0.001225, 5)
        g = noisy.astype(np.float64) / 255
        P = np.pad(g, 2, mode="edge")
        from scipy import ndimage
        m = ndimage.uniform_filter(g, 5, mode="nearest"); v = ndimage.uniform_filter(g * g, 5, mode="nearest") - m * m
        want = np.clip(np.rint((g - np.minimum(0.001225 / np.maximum(v, 1e-12), 1) * (g - m)) * 255), 0, 255).astype(np.uint8)
        ensure(np.abs(got.astype(int) - want.astype(int)).max() <= 1, "formül")
        rows = [["gürültülü", f"{cv2.PSNR(clean, noisy):.2f}"], ["uyarlamalı ortalama", f"{cv2.PSNR(clean, got):.2f}"],
                ["kutu 5×5", f"{cv2.PSNR(clean, cv2.blur(noisy, (5, 5))):.2f}"], ["Gauss 5×5", f"{cv2.PSNR(clean, cv2.GaussianBlur(noisy, (5, 5), 0)):.2f}"]]
        R.table("Gauss gürültüsü σ² = 0.001225 (alıştırma 4)", ["çıkış", "PSNR (dB)"], rows,
                note="σ_η²'yi yanlış (ör. 0.009) verirsen oran çoğu yerde 1'e kırpılır ve filtre kutu filtresine dönüşür: dene.")
    R.test("adaptive_mean (alıştırma 4)", am)
