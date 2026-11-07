import numpy as np
import cv2
from skimage import data
from labcheck import close, no_loops, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("döngü yok", lambda: no_loops(src))
    img = data.camera()
    f = img.astype(np.float32)

    def bs():
        for s in (3, 11, 21):
            close(need(ns, "box_separable")(img, s), cv2.boxFilter(f, -1, (s, s), normalize=True, borderType=cv2.BORDER_CONSTANT), atol=1e-2, what=f"{s}×{s}")
    R.test("box_separable = cv2.boxFilter (3, 11, 21)", bs)

    def ii():
        A = need(ns, "integral")(img)
        ensure(A.shape == (513, 513) and np.array_equal(A, cv2.integral(img)), "cv2.integral ile birebir")
    R.test("integral = cv2.integral", ii)

    def bi():
        for s in (3, 21):
            got = need(ns, "box_integral")(img, s)
            close(got, cv2.boxFilter(f, -1, (s, s), normalize=True, borderType=cv2.BORDER_CONSTANT), atol=1e-2, what=f"{s}×{s}")
        u8 = np.rint(ns["box_integral"](img, 3)).astype(np.uint8)
        ensure(np.abs(u8.astype(int) - cv2.blur(img, (3, 3), borderType=cv2.BORDER_CONSTANT).astype(int)).max() <= 1, "uint8 ±1")
    R.test("box_integral (4 arama) = cv2", bi)

    def gk():
        for k in (3, 7, 15):
            close(need(ns, "gaussian_kernel")(k), cv2.getGaussianKernel(k, 0).ravel(), atol=1e-7, what=f"ksize {k}, σ otomatik")
        close(ns["gaussian_kernel"](9, 2.5), cv2.getGaussianKernel(9, 2.5).ravel(), atol=1e-7, what="σ = 2.5")
    R.test("gaussian_kernel = cv2.getGaussianKernel", gk)

    def gb():
        for k, s in ((7, 0), (15, 3.0)):
            close(need(ns, "gaussian_blur")(img, k, s), cv2.GaussianBlur(f, (k, k), s), atol=1e-2, what=f"ksize {k}")
    R.test("gaussian_blur = cv2.GaussianBlur (REFLECT_101)", gb)
