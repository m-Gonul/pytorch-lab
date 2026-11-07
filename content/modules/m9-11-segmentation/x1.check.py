import numpy as np
import cv2
from skimage import data
from labcheck import need, ensure


def check(ns, R):
    rows = []
    for name in ("camera", "coins", "moon", "page", "text"):
        def t(name=name):
            img = getattr(data, name)()
            want, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            got = need(ns, "otsu")(img)
            ensure(got == int(want), f"{got}, OpenCV {int(want)}")
            rows.append([name, got])
        R.test(f"otsu = cv2 ({name})", t)
    if rows:
        R.table("Otsu eşikleri", ["görüntü", "eşik"], rows)

    def ad():
        img = data.page()
        for b, C in [(11, 2), (25, 10), (5, 0), (15, -3)]:
            ensure(np.array_equal(need(ns, "adaptive_mean")(img, b, C), cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, b, C)), f"blok {b}, C {C}")
        t = ns["otsu"](img)
        g = (img > t).astype(np.uint8) * 255
        return f"'page' (eşit olmayan aydınlatma): Otsu sayfanın alt kısmını kaybeder; uyarlamalı eşik metni her yerde ayırır"
    R.test("adaptive_mean = cv2.adaptiveThreshold (MEAN_C)", ad)
