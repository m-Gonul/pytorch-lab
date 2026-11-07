import numpy as np
import cv2
from scipy import ndimage
from skimage import data
from labcheck import close, no_loops, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("döngü yok", lambda: no_loops(src))
    img = data.camera()
    ses = {"kare 5": np.ones((5, 5), bool), "elips 7": cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)).astype(bool), "artı 3": np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)}
    ops = {"opening": cv2.MORPH_OPEN, "closing": cv2.MORPH_CLOSE, "gradient": cv2.MORPH_GRADIENT, "tophat": cv2.MORPH_TOPHAT, "blackhat": cv2.MORPH_BLACKHAT}
    for name, se in ses.items():
        def t(se=se):
            k = se.astype(np.uint8)
            ensure(np.array_equal(need(ns, "gray_erode")(img, se), cv2.erode(img, k)), "erozyon")
            ensure(np.array_equal(need(ns, "gray_dilate")(img, se), cv2.dilate(img, k)), "genişleme")
            for fn, op in ops.items():
                ensure(np.array_equal(need(ns, fn)(img, se), cv2.morphologyEx(img, op, k)), fn)
        R.test(f"7 işlem = cv2 ({name})", t)

    def asym():
        se = np.array([[0, 1, 0], [1, 1, 1], [0, 0, 1]], bool)
        k = se.astype(np.uint8)
        mine = need(ns, "opening")(img, se)
        cvo = cv2.morphologyEx(img, cv2.MORPH_OPEN, k)
        ensure(np.array_equal(ns["gray_dilate"](img, se), cv2.dilate(img, k[::-1, ::-1].copy())), "asimetrik genişleme = cv2.dilate(yansıtılmış)")
        ensure(np.all(mine <= img), "açma görüntüden büyük olamaz (anti-ekstansif)")
        ensure(np.array_equal(ns["opening"](mine, se), mine), "açma idempotent olmalı")
        return f"asimetrik SE: cv2 MORPH_OPEN ile {int((mine != cvo).sum())} piksel farklı; ders kitabı açması anti-ekstansif ve idempotent"
    R.test("asimetrik SE ve açmanın özellikleri", asym)

    def nf():
        bN = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
        got = need(ns, "nonflat_erode")(img, bN)
        want = ndimage.grey_erosion(img.astype(np.float64), structure=bN.astype(np.float64), mode="constant", cval=np.inf)
        close(got, want, what="düz olmayan erozyon")
        close(ns["nonflat_erode"](img, np.zeros((3, 3))), cv2.erode(img, np.ones((3, 3), np.uint8)).astype(np.float64), what="bN = 0 → düz erozyon")
    R.test("nonflat_erode", nf)
