import numpy as np
import cv2
from skimage import data
from labcheck import close, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("cv2/numpy kullanılmıyor", lambda: no_imports(src, {"cv2", "numpy", "scipy", "skimage"}))
    R.test("bilinear_at (elle örnek)", lambda: close(need(ns, "bilinear_at")([[10.0, 20.0], [30.0, 40.0]], 0.25, 0.5), 22.5))
    R.test("bilinear_at: kenar kırpma", lambda: close(need(ns, "bilinear_at")([[10.0, 20.0], [30.0, 40.0]], 5.0, -3.0), 20.0))
    crop = data.camera()[200:224, 240:272].astype(np.float32)
    L = crop.tolist()

    def nn():
        for H2, W2 in ((12, 16), (40, 50)):
            got = np.array(need(ns, "resize_nearest")(L, H2, W2), np.float32)
            close(got, cv2.resize(crop, (W2, H2), interpolation=cv2.INTER_NEAREST), what=f"{H2}×{W2}")
    R.test("resize_nearest = cv2.INTER_NEAREST", nn)

    def bil():
        for H2, W2 in ((10, 14), (48, 60), (24, 32)):
            got = np.array(need(ns, "resize_bilinear")(L, H2, W2), np.float32)
            close(got, cv2.resize(crop, (W2, H2), interpolation=cv2.INTER_LINEAR), atol=1e-3, what=f"{H2}×{W2}")
    R.test("resize_bilinear = cv2.INTER_LINEAR (küçültme, büyütme, aynı boyut)", bil)
