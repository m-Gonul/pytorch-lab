import numpy as np
import cv2
from skimage import data, color
from labcheck import close, need, ensure


def check(ns, R):
    def rec():
        for x in (data.camera().astype(np.float64), data.coins().astype(np.float64)[:301, :383]):
            close(need(ns, "collapse")(need(ns, "laplacian_pyramid")(x, 5)), x, atol=1e-9, what=f"{x.shape} yeniden kurma")
        G = need(ns, "gaussian_pyramid")(data.camera(), 4)
        ensure([g.shape for g in G] == [(512, 512), (256, 256), (128, 128), (64, 64)], "piramit boyutları")
    R.test("Laplacian piramidi kayıpsız", rec)

    def bl():
        a = data.camera().astype(np.float64)
        b = cv2.resize(color.rgb2gray(data.astronaut()) * 255, (512, 512))
        b = b * 0.7 + 60
        mask = np.zeros((512, 512)); mask[:, :256] = 1
        hard = a * mask + b * (1 - mask)
        soft = need(ns, "blend")(a, b, mask, 6)
        jump = lambda im: np.abs(im[:, 257] - im[:, 254]).mean()
        ensure(jump(soft) < jump(hard) * 0.6, f"dikişteki sıçrama: sert {jump(hard):.1f}, harmanlanmış {jump(soft):.1f}")
        far = np.abs(soft[:, :200] - a[:, :200]).mean()
        ensure(far < 3, "dikişten uzakta görüntü korunmalı")
        return f"dikişte ortalama sıçrama: sert kesim {jump(hard):.1f}, çok bantlı {jump(soft):.1f}; dikişten uzakta değişim {far:.2f}"
    R.test("çok bantlı harmanlama", bl)
