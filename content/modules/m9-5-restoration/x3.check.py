import numpy as np
import cv2
from skimage import data
from skimage.metrics import structural_similarity
from labcheck import close, need, ensure


def check(ns, R):
    clean = data.camera()

    def table():
        rows = []
        for p in (0.05, 0.2, 0.4):
            rng = np.random.default_rng(int(p * 100))
            noisy = clean.copy(); u = rng.random(clean.shape); noisy[u < p / 2] = 0; noisy[(u >= p / 2) & (u < p)] = 255
            t = need(ns, "denoise_table")(clean, noisy)
            ensure(set(t) == {"box5", "gauss7", "median5", "cwmedian5"}, "dört anahtar")
            close(t["median5"], cv2.PSNR(clean, cv2.medianBlur(noisy, 5)), atol=1e-6, what="median5")
            rows.append([f"%{int(p * 100)}"] + [f"{t[k]:.2f}" for k in ("box5", "gauss7", "median5", "cwmedian5")])
        R.table("Tuz-biber, PSNR (dB)", ["gürültü", "kutu 5", "Gauss 7", "median 5", "CW median 5"], rows,
                note="Düşük yoğunlukta merkez ağırlıklı median ayrıntıyı daha iyi korur; yüksek yoğunlukta merkezdeki gürültüyü de koruduğu için geride kalır.")
    R.test("filtre karşılaştırma tablosu", table)

    def fool():
        out = need(ns, "fool_psnr")(clean)
        p = cv2.PSNR(clean, out)
        s = structural_similarity(clean, out, data_range=255)
        ensure(out.shape == clean.shape and out.dtype == np.uint8, "aynı boyut, uint8")
        ensure(p < 25 and s > 0.9, f"PSNR {p:.2f} dB, SSIM {s:.3f}: PSNR < 25 ve SSIM > 0.9 bekleniyordu")
        return f"PSNR {p:.2f} dB, SSIM {s:.3f}: görsel olarak aynı, PSNR'a göre 'kötü'"
    R.test("PSNR'ı kandır (SSIM yüksek kalsın)", fool)
