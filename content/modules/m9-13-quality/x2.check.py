import numpy as np
import cv2
from skimage import data, color
from skimage.metrics import structural_similarity as sk
from labcheck import close, need, ensure


def check(ns, R):
    hr = data.astronaut()
    lr = cv2.resize(hr, (128, 128), interpolation=cv2.INTER_AREA)
    up = cv2.resize(lr, (512, 512), interpolation=cv2.INTER_CUBIC)

    def y():
        close(need(ns, "rgb_to_y")(hr), color.rgb2ycbcr(hr)[..., 0], atol=1e-9, what="skimage rgb2ycbcr Y")
    R.test("rgb_to_y = skimage.color.rgb2ycbcr", y)

    def ps():
        p0, p4 = need(ns, "psnr_y")(hr, up), ns["psnr_y"](hr, up, 4)
        rgb = 10 * np.log10(255 ** 2 / ((hr.astype(float) - up) ** 2).mean())
        s = need(ns, "ssim_rgb")(hr, up)
        close(s, sk(hr.astype(float), up.astype(float), channel_axis=2, data_range=255, gaussian_weights=True, sigma=1.5, use_sample_covariance=False), atol=1e-10, what="kanal ortalamalı SSIM")
        R.table("4× bikübik büyütme (astronaut)", ["ölçüt", "değer"], [["PSNR (RGB)", f"{rgb:.2f}"], ["PSNR (Y)", f"{p0:.2f}"], ["PSNR (Y, kenar 4 kırpık)", f"{p4:.2f}"], ["SSIM (RGB kanal ortalaması)", f"{s:.4f}"]],
                note="Aynı çıktı için dört farklı 'PSNR': makaleler arası karşılaştırmada protokolü mutlaka kontrol et.")
    R.test("psnr_y ve ssim_rgb", ps)
