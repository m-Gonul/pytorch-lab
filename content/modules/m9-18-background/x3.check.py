import numpy as np
import cv2
from skimage import data
from labcheck import need, ensure


def check(ns, R):
    bg = cv2.resize(data.coffee(), (200, 150))
    img = bg.copy()
    img[40:100, 30:80] = (np.array([20, 120, 220]) * np.ones((60, 50, 1))).astype(np.uint8)
    sh = np.zeros(bg.shape[:2], bool); sh[90:130, 80:160] = True
    img[sh] = (img[sh].astype(np.float64) * 0.6).astype(np.uint8)
    obj = np.zeros_like(sh); obj[40:100, 30:80] = True
    fg = obj | sh

    def t():
        m = need(ns, "shadow_mask")(img, bg, fg)
        rec = (m & sh & ~obj).sum() / (sh & ~obj).sum()
        fa = (m & obj).sum() / obj.sum()
        ensure(rec > 0.8 and fa < 0.1, f"gölge yakalama {rec:.2f}, nesneyi gölge sanma {fa:.2f}")
        return f"gölge piksellerinin %{rec * 100:.0f}'ı gölge, nesne piksellerinin %{fa * 100:.0f}'ı yanlışlıkla gölge"
    R.test("shadow_mask", t)
