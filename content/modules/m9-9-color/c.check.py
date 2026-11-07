import numpy as np
import cv2
from skimage import data
from labcheck import close, no_loops, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("döngü yok", lambda: no_loops(src))
    rng = np.random.default_rng(0)
    imgs = {"astronaut": data.astronaut(), "coffee": data.coffee(), "rastgele": rng.integers(0, 256, (128, 128, 3)).astype(np.uint8),
            "gri ve uçlar": np.array([[[0, 0, 0], [255, 255, 255], [128, 128, 128], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 0, 255], [1, 0, 0]]], np.uint8)}
    for name, im in imgs.items():
        def t(im=im):
            got = need(ns, "rgb2hsv8")(im); want = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
            ensure(np.array_equal(got, want), f"HSV: {(got != want).any(-1).sum()} piksel farklı")
            got = need(ns, "rgb2ycrcb8")(im); want = cv2.cvtColor(im, cv2.COLOR_RGB2YCrCb)
            ensure(np.array_equal(got, want), f"YCrCb: {(got != want).any(-1).sum()} piksel farklı")
        R.test(f"= cv2.cvtColor ({name})", t)

    def hsi():
        im = imgs["rastgele"]
        got = need(ns, "rgb2hsi")(im)
        f = im.astype(np.float64) / 255
        close(got[..., 2], f.mean(-1), what="I")
        close(got[..., 1], 1 - 3 * f.min(-1) / f.sum(-1), atol=1e-12, what="S")
        hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV_FULL).astype(np.float64)[..., 0] * 360 / 256
        d = np.abs((got[..., 0] - hsv + 180) % 360 - 180)
        ensure(np.median(d) < 6, f"HSI tonu HSV tonundan çok farklı: medyan {np.median(d):.1f}°")
        return f"HSI tonu ile HSV tonu arasında medyan fark {np.median(d):.1f}° (altıgen ve dairesel ton tanımları farklı)"
    R.test("rgb2hsi", hsi)
