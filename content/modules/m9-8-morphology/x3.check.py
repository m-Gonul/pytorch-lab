import numpy as np
import cv2
from skimage import data
from labcheck import close, need, ensure


def disks_image(radii, seed=0):
    rng = np.random.default_rng(seed)
    img = np.zeros((256, 256), np.uint8)
    for r in radii:
        for _ in range(6):
            cv2.circle(img, (int(rng.integers(r, 256 - r)), int(rng.integers(r, 256 - r))), r, 200, -1)
    return img


def check(ns, R):
    radii = list(range(1, 24))

    def gran():
        img = disks_image([12])
        g = need(ns, "granulometry")(img, radii)
        ensure(len(g) == len(radii) and g[0] > g[-1], "açma büyüdükçe toplam azalmalı")
        ps = need(ns, "pattern_spectrum")(img, radii)
        close(ps.sum(), 1.0, what="toplam 1")
        peak = radii[int(np.argmax(ps)) + 1]
        ensure(abs(peak - 13) <= 1, f"tepe yarıçapı {peak}, ≈ 12–13 bekleniyordu")
        two = ns["pattern_spectrum"](disks_image([5, 15], 1), radii)
        top2 = sorted(radii[i + 1] for i in np.argsort(two)[-2:])
        return f"tek boyut: tepe r = {peak}; iki boyut: tepeler r = {top2}"
    R.test("granülometri ve desen spektrumu", gran)

    def sm():
        clean = data.camera()
        rng = np.random.default_rng(0)
        noisy = clean.copy(); u = rng.random(clean.shape); noisy[u < 0.03] = 0; noisy[u > 0.97] = 255
        out = need(ns, "oc_smooth")(noisy, 1)
        p0, p1 = cv2.PSNR(clean, noisy), cv2.PSNR(clean, out)
        ensure(p1 > p0 + 5, f"PSNR {p0:.1f} → {p1:.1f}")
        return f"tuz-biber %6: PSNR {p0:.1f} → {p1:.1f} dB (median 3×3: {cv2.PSNR(clean, cv2.medianBlur(noisy, 3)):.1f})"
    R.test("açma–kapama yumuşatma", sm)
