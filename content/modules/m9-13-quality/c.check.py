import numpy as np
from skimage import data
from skimage.metrics import structural_similarity as sk_ssim
from labcheck import close, need, ensure


def check(ns, R):
    img = data.camera()

    def eq():
        rng = np.random.default_rng(0)
        for y in (np.clip(img + rng.normal(0, 20, img.shape), 0, 255), img[::-1].astype(float), img * 0.5 + 40):
            want = sk_ssim(img.astype(float), y.astype(float), data_range=255, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
            close(need(ns, "ssim")(img, y), want, atol=1e-10, what="skimage SSIM")
    R.test("ssim = skimage structural_similarity", eq)

    def fig():
        target = 210.0
        rows = []
        res = {}
        for name in ("mean_shift", "contrast", "noise", "impulse", "blur", "jpeg"):
            out = need(ns, name)(img, target)
            m = float(((out - img.astype(float)) ** 2).mean())
            tol = 0.15 if name == "jpeg" else 0.03
            ensure(abs(m - target) / target < tol, f"{name}: MSE {m:.1f}")
            res[name] = ns["ssim"](img, out)
            rows.append([name, f"{m:.1f}", f"{res[name]:.3f}"])
        ensure(res["mean_shift"] > 0.9 and res["mean_shift"] == max(res.values()), "ortalama kayması en yüksek SSIM'i almalı")
        ensure(res["blur"] < 0.8 and res["jpeg"] < 0.8 and res["noise"] < 0.8, "bulanıklık, JPEG ve gürültü düşük SSIM almalı")
        R.table("Aynı MSE (≈ 210), çok farklı kalite (Wang 2004, Şekil 2)", ["bozulma", "MSE", "SSIM"], rows,
                note="PSNR hepsine ≈ 24.9 dB der; gözünüz ve SSIM katılmaz.")
    R.test("eşit MSE figürü", fig)
