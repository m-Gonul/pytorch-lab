import numpy as np
import cv2
from skimage import data
from labcheck import need, ensure


def check(ns, R):
    rng = np.random.default_rng(0)
    img = np.clip(data.astronaut()[100:228, 150:278] + rng.normal(0, 12, (128, 128, 3)), 0, 255).astype(np.uint8)

    def col():
        for d, sc, ss in [(7, 30, 5), (9, 60, 3)]:
            got = need(ns, "bilateral_color")(img, d, sc, ss)
            want = cv2.bilateralFilter(img, d, sc, ss)
            diff = np.abs(got.astype(int) - want)
            ensure(diff.max() <= 1 and (diff > 0).mean() < 0.01, f"d = {d}: en büyük fark {diff.max()}, %{(diff > 0).mean() * 100:.2f}")
    R.test("bilateral_color ≈ cv2.bilateralFilter (3 kanal)", col)

    def joint():
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        a = need(ns, "joint_bilateral")(g, g, 9, 40, 3)
        b = cv2.bilateralFilter(g, 9, 40, 3)
        ensure(np.abs(a.astype(int) - b).max() <= 1, "rehber = görüntü iken bilateral ile aynı olmalı")
        clean = cv2.cvtColor(data.astronaut()[100:228, 150:278], cv2.COLOR_RGB2GRAY)
        noisy = np.clip(clean + rng.normal(0, 25, clean.shape), 0, 255).astype(np.uint8)
        j = ns["joint_bilateral"](noisy, clean, 9, 20, 3)
        s = cv2.bilateralFilter(noisy, 9, 60, 3)
        pj, ps = cv2.PSNR(clean, j), cv2.PSNR(clean, s)
        ensure(pj > ps, "temiz rehberle ortak bilateral daha iyi olmalı")
        return f"σ = 25 gürültü: bilateral {ps:.2f} dB, temiz rehberle ortak bilateral {pj:.2f} dB (flaş/flaşsız fikri)"
    R.test("joint_bilateral", joint)
