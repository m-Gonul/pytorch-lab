import numpy as np
import cv2


def shadow_mask(img, bg, fg, alpha=0.4, beta=0.93, tS=0.15, tH=40):
    hi = cv2.cvtColor(img, cv2.COLOR_RGB2HSV_FULL).astype(np.float64)
    hb = cv2.cvtColor(bg, cv2.COLOR_RGB2HSV_FULL).astype(np.float64)
    r = hi[..., 2] / np.maximum(hb[..., 2], 1)
    dS = np.abs(hi[..., 1] - hb[..., 1]) / 255
    dH = np.abs(hi[..., 0] - hb[..., 0]) * 360 / 256
    dH = np.minimum(dH, 360 - dH)
    return fg & (r >= alpha) & (r <= beta) & (dS <= tS) & (dH <= tH)


print("gölge")
