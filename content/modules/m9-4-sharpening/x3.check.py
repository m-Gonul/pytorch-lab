import numpy as np
import cv2
from skimage import data
from labcheck import close, need, ensure


def check(ns, R):
    def k3():
        K = need(ns, "cv_laplacian3")()
        f = data.camera().astype(np.float32)
        close(cv2.filter2D(f, -1, np.asarray(K, np.float32)), cv2.Laplacian(f, cv2.CV_32F, ksize=3), atol=1e-2)
        return "cv2.Laplacian(ksize=3) çekirdeği:\n" + np.array2string(np.asarray(K))
    R.test("deneyle bulunan çekirdek cv2.Laplacian(ksize=3) ile aynı sonucu veriyor", k3)

    def disp():
        f = data.camera().astype(np.float32)
        L = cv2.Laplacian(f, cv2.CV_32F, ksize=1)
        full, mid = need(ns, "laplacian_display")(L)
        ensure(full.dtype == np.uint8 and full.min() == 0 and full.max() == 255, "tam aralık")
        ensure(mid.dtype == np.uint8 and abs(int(mid[np.abs(L) < 1e-6].mean()) - 128) <= 1, "sıfır → 128")
    R.test("laplacian_display", disp)
