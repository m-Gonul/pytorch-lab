import numpy as np
import cv2
from skimage import data
from labcheck import close, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("cv2/numpy kullanılmıyor", lambda: no_imports(src, {"cv2", "numpy", "skimage", "scipy"}))
    f = [5, 5, 4, 3, 2, 1, 0, 0, 0, 6, 0, 0]

    def d():
        ensure(list(need(ns, "diff1")(f)) == [0, -1, -1, -1, -1, -1, 0, 0, 6, -6, 0], "diff1")
        ensure(list(need(ns, "diff2")(f)) == [-1, 0, 0, 0, 0, 1, 0, 6, -12, 6], "diff2")
    R.test("diff1/diff2 (dersteki profil)", d)
    crop = data.camera()[300:340, 50:96].astype(np.float32)

    def lap():
        got = np.array(need(ns, "laplacian")(crop.tolist()), np.float32)
        close(got, cv2.Laplacian(crop, cv2.CV_32F, ksize=1), atol=1e-3)
    R.test("laplacian = cv2.Laplacian(ksize=1), REFLECT_101", lap)

    def sob():
        gx, gy = need(ns, "sobel")(crop.tolist())
        close(np.array(gx, np.float32), cv2.Sobel(crop, cv2.CV_32F, 1, 0, ksize=3), atol=1e-3, what="gx")
        close(np.array(gy, np.float32), cv2.Sobel(crop, cv2.CV_32F, 0, 1, ksize=3), atol=1e-3, what="gy")
    R.test("sobel = cv2.Sobel(ksize=3)", sob)
