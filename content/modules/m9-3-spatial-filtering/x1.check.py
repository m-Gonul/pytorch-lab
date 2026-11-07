import numpy as np
import cv2
from skimage import data
from labcheck import close, need, ensure, uses_any


def check(ns, R):
    src = ns["__source__"]
    R.test("np.pad / copyMakeBorder kullanılmıyor", lambda: ensure("np.pad" not in src and "copyMakeBorder" not in src, "kendi indekslerinle yaz"))
    img = data.camera()[:40, :50]
    modes = {"constant": cv2.BORDER_CONSTANT, "replicate": cv2.BORDER_REPLICATE, "reflect": cv2.BORDER_REFLECT,
             "reflect101": cv2.BORDER_REFLECT_101, "wrap": cv2.BORDER_WRAP}
    for name, bt in modes.items():
        def t(name=name, bt=bt):
            for r in (1, 3, 7):
                got = need(ns, "pad")(img, r, name)
                ensure(np.array_equal(got, cv2.copyMakeBorder(img, r, r, r, r, bt, value=0)), f"{name}, r = {r}")
        R.test(f"{name}", t)
