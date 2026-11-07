import numpy as np
import cv2
from skimage import data
from labcheck import close, need, ensure


def check(ns, R):
    rng = np.random.default_rng(0)
    img = rng.random((50, 60, 3))

    def cube():
        a = np.array([0.6, 0.3, 0.2])
        got = need(ns, "slice_cube")(img, a, 0.4)
        inside = (np.abs(img - a) <= 0.2).all(-1)
        ensure(np.array_equal(got[inside], img[inside]) and np.all(got[~inside] == 0.5), "küp")
        got = need(ns, "slice_sphere")(img, a, 0.3)
        inside = ((img - a) ** 2).sum(-1) <= 0.09
        ensure(np.array_equal(got[inside], img[inside]) and np.all(got[~inside] == 0.5), "küre")
        ensure(not np.shares_memory(got, img), "girdi değiştirilmemeli")
    R.test("renk dilimleme (küp ve küre)", cube)

    def eq():
        im = (data.coffee() * 0.55).astype(np.uint8)
        a = need(ns, "equalize_intensity")(im)
        b = need(ns, "equalize_channels")(im)
        ensure(a.shape == im.shape and a.dtype == np.uint8, "uint8, aynı boyut")
        h0 = ns["rgb2hsi"](im / 255.0); ha = ns["rgb2hsi"](a / 255.0); hb = ns["rgb2hsi"](b / 255.0)
        sat = h0[..., 1] > 0.15
        dh = lambda h: np.median(np.abs((h[..., 0] - h0[..., 0] + 180) % 360 - 180)[sat])
        ensure(dh(ha) < 3, f"I eşitlemede ton {dh(ha):.1f}° değişti")
        ensure(dh(hb) > dh(ha) + 2, "kanal kanal eşitleme tonu daha çok bozmalı")
        R.table("Koyu kahve görüntüsü", ["yöntem", "ort. I", "medyan ton kayması"], [["girdi", f"{h0[..., 2].mean():.2f}", "—"], ["yalnız I eşitle (HSI)", f"{ha[..., 2].mean():.2f}", f"{dh(ha):.1f}°"], ["R, G, B ayrı eşitle", f"{hb[..., 2].mean():.2f}", f"{dh(hb):.1f}°"]])
    R.test("HSI'da yoğunluk eşitleme tonu korur", eq)
