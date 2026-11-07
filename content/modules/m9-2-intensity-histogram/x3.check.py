import numpy as np
import cv2
from skimage import data
from labcheck import close, no_loops, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("döngü yok", lambda: no_loops(src))
    img = data.moon()

    def st():
        got = need(ns, "stretch")(img)
        ref = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        ensure(got.dtype == np.uint8 and np.abs(got.astype(int) - ref.astype(int)).max() <= 1, "cv2.normalize ile ±1")
        ensure(got.min() == 0 and got.max() == 255, "tam aralık")
    R.test("stretch = cv2.normalize(NORM_MINMAX)", st)

    def pw():
        r = np.arange(256, dtype=np.uint8)
        y = need(ns, "piecewise")(r, 80, 20, 160, 235)
        ensure(int(y[80]) == 20 and int(y[160]) == 235 and int(y[0]) == 0 and int(y[255]) == 255, "düğüm noktaları")
        ensure(bool(np.all(np.diff(y.astype(int)) >= 0)), "monoton")
        ensure(abs(int(y[120]) - round(20 + 215 * 40 / 80)) <= 0, "orta parça doğrusal")
    R.test("piecewise", pw)

    def sl():
        b = need(ns, "slice_level")(img, 100, 150, "binary")
        ensure(set(np.unique(b)) <= {0, 255} and np.array_equal(b == 255, (img >= 100) & (img <= 150)), "binary")
        p = ns["slice_level"](img, 100, 150, "preserve")
        ensure(np.array_equal(p[(img < 100)], img[img < 100]) and np.all(p[(img >= 100) & (img <= 150)] == 255), "preserve")
    R.test("slice_level (iki mod)", sl)
