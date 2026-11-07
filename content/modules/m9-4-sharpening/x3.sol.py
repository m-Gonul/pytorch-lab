import numpy as np
import cv2


def cv_laplacian3():
    imp = np.zeros((5, 5), np.float32)
    imp[2, 2] = 1
    out = cv2.Laplacian(imp, cv2.CV_32F, ksize=3)
    return out[1:4, 1:4][::-1, ::-1].copy()


def laplacian_display(L):
    L = np.asarray(L, np.float64)
    lo, hi = L.min(), L.max()
    full = np.rint((L - lo) * 255 / (hi - lo)) if hi > lo else np.zeros_like(L)
    m = np.abs(L).max()
    mid = np.rint(128 + L * 127 / m) if m > 0 else np.full_like(L, 128)
    return full.astype(np.uint8), mid.astype(np.uint8)


print(cv_laplacian3())
