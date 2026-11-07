import numpy as np
import cv2
from skimage import data
from labcheck import close, no_loops, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("döngü yok", lambda: no_loops(src))
    ims = {"camera": data.camera(), "moon": data.moon(), "coins": data.coins(), "page": data.page()}

    def eq():
        for name, im in ims.items():
            got = need(ns, "equalize")(im, "opencv")
            ensure(got.dtype == np.uint8 and np.array_equal(got, cv2.equalizeHist(im)), f"{name}: cv2 ile farklı")
    R.test("equalize('opencv') = cv2.equalizeHist (4 görüntü)", eq)

    def eqc():
        im = ims["moon"]
        c = np.bincount(im.ravel(), minlength=256).cumsum()
        ensure(np.array_equal(need(ns, "equalize")(im, "class"), np.rint(255 * c / im.size).astype(np.uint8)[im]), "ders formülü")
    R.test("equalize('class')", eqc)

    def pts():
        im = ims["camera"]
        ensure(np.array_equal(need(ns, "negative")(im), 255 - im), "negatif")
        r = np.arange(256, dtype=np.uint8).reshape(16, 16)
        close(need(ns, "log_transform")(r).astype(float), np.rint(255 / np.log(256) * np.log1p(r.astype(float))), what="log")
        close(need(ns, "gamma")(r, 0.5).astype(float), np.rint(255 * (r / 255) ** 0.5), what="gamma 0.5")
        ensure(need(ns, "gamma")(r, 1.0).tolist() == r.tolist(), "gamma 1 özdeşlik")
        g2 = ns["gamma"](im, 2.2)
        ensure(g2.dtype == np.uint8 and g2.mean() < im.mean(), "γ > 1 koyulaştırır")
    R.test("negatif, log, gamma", pts)
