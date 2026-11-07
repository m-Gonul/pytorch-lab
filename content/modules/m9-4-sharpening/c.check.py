import numpy as np
import cv2
from skimage import data
from labcheck import close, no_loops, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("döngü yok", lambda: no_loops(src))
    img = data.moon()
    f = img.astype(np.float32)

    def sl():
        for kind, K in ((4, np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32)), (8, np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], np.float32))):
            want = np.clip(np.rint(f - cv2.filter2D(f, -1, K)), 0, 255).astype(np.uint8)
            got = need(ns, "sharpen_laplacian")(img, kind)
            ensure(got.dtype == np.uint8 and np.abs(got.astype(int) - want.astype(int)).max() <= 1, f"kind = {kind}")
    R.test("sharpen_laplacian (4 ve 8 komşu)", sl)

    def us():
        for k in (1.0, 2.5):
            blur = cv2.GaussianBlur(f, (5, 5), 0)
            want = np.clip(np.rint(f + k * (f - blur)), 0, 255).astype(np.uint8)
            got = need(ns, "unsharp")(img, k)
            ensure(np.abs(got.astype(int) - want.astype(int)).max() <= 1, f"k = {k}")
    R.test("unsharp ve highboost", us)

    def gm():
        gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
        close(need(ns, "gradient_magnitude")(img, "l2"), cv2.magnitude(gx, gy), atol=1e-2, what="L2")
        close(ns["gradient_magnitude"](img, "l1"), np.abs(gx) + np.abs(gy), atol=1e-2, what="L1")
    R.test("gradient_magnitude = cv2.magnitude(Sobel)", gm)
