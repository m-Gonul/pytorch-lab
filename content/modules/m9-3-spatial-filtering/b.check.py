import numpy as np
import cv2
from skimage import data
from labcheck import close, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("cv2/numpy kullanılmıyor", lambda: no_imports(src, {"cv2", "numpy", "skimage", "scipy"}))
    crop = data.camera()[150:190, 200:236].astype(np.float32)
    rng = np.random.default_rng(0)
    K = rng.normal(size=(5, 3)).astype(np.float32)
    K[0, 0] += 2

    def corr():
        got = np.array(need(ns, "correlate2d")(crop.tolist(), K.tolist()), np.float32)
        close(got, cv2.filter2D(crop, -1, K, borderType=cv2.BORDER_CONSTANT), atol=1e-3)
    R.test("correlate2d = cv2.filter2D (5 × 3 asimetrik çekirdek)", corr)

    def conv():
        got = np.array(need(ns, "convolve2d")(crop.tolist(), K.tolist()), np.float32)
        close(got, cv2.filter2D(crop, -1, K[::-1, ::-1].copy(), borderType=cv2.BORDER_CONSTANT), atol=1e-3)
        imp = [[0.0] * 7 for _ in range(7)]; imp[3][3] = 1.0
        close(np.array(ns["convolve2d"](imp, K.tolist()))[1:6, 2:5], K, atol=1e-6, what="dürtü yanıtı = çekirdek")
    R.test("convolve2d: çevrilmiş çekirdek, dürtü yanıtı", conv)

    def box():
        for s in (3, 11):
            got = np.array(need(ns, "box_filter")(crop.tolist(), s), np.float32)
            close(got, cv2.boxFilter(crop, -1, (s, s), normalize=True, borderType=cv2.BORDER_CONSTANT), atol=1e-3, what=f"{s}×{s}")
        u8 = np.rint(np.array(ns["box_filter"](crop.tolist(), 3))).astype(np.uint8)
        ref = cv2.blur(crop.astype(np.uint8), (3, 3), borderType=cv2.BORDER_CONSTANT)
        ensure(np.abs(u8.astype(int) - ref.astype(int)).max() <= 1, "uint8'de ±1 (alıştırmadaki ≤3 sınırının altında)")
    R.test("box_filter = cv2.boxFilter (sıfır doldurma)", box)
