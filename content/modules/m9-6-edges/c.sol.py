import numpy as np
import cv2
from scipy import ndimage


def canny_cv(img, lo, hi, l2=False):
    if lo > hi:
        lo, hi = hi, lo
    dx = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3, borderType=cv2.BORDER_REPLICATE).astype(np.int64)
    dy = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=3, borderType=cv2.BORDER_REPLICATE).astype(np.int64)
    if l2:
        m = dx * dx + dy * dy
        lo, hi = int(np.floor(lo * lo)), int(np.floor(hi * hi))
    else:
        m = np.abs(dx) + np.abs(dy)
        lo, hi = int(np.floor(lo)), int(np.floor(hi))
    H, W = img.shape
    P = np.pad(m, 1)
    sh = lambda a, b: P[1 + a:1 + a + H, 1 + b:1 + b + W]
    ax, ay = np.abs(dx), np.abs(dy) << 15
    tg22x = ax * 13573
    tg67x = tg22x + (ax << 16)
    horiz = ay < tg22x
    vert = ~horiz & (ay > tg67x)
    diag = ~(horiz | vert)
    pos = (dx ^ dy) >= 0
    d1 = np.where(pos, sh(-1, -1), sh(-1, 1))
    d2 = np.where(pos, sh(1, 1), sh(1, -1))
    keep = (horiz & (m > sh(0, -1)) & (m >= sh(0, 1))) | (vert & (m > sh(-1, 0)) & (m >= sh(1, 0))) | (diag & (m > d1) & (m > d2))
    cand = keep & (m > lo)
    lab, n = ndimage.label(cand, np.ones((3, 3)))
    ok = np.zeros(n + 1, bool)
    ok[np.unique(lab[cand & (m > hi)])] = True
    ok[0] = False
    return (ok[lab] * 255).astype(np.uint8)


def marr_hildreth(img, sigma, thr_frac=0.04):
    n = int(np.ceil(6 * sigma))
    n += 1 - n % 2
    g = cv2.Laplacian(cv2.GaussianBlur(img.astype(np.float64), (n, n), sigma), cv2.CV_64F)
    thr = thr_frac * g.max()
    H, W = g.shape
    c = lambda a, b: g[1 + a:H - 1 + a, 1 + b:W - 1 + b]
    zc = lambda p, q: (p * q < 0) & (np.abs(p - q) > thr)
    z = np.zeros((H, W), bool)
    z[1:-1, 1:-1] = zc(c(0, -1), c(0, 1)) | zc(c(-1, 0), c(1, 0)) | zc(c(-1, -1), c(1, 1)) | zc(c(-1, 1), c(1, -1))
    return z


from skimage import data
print((canny_cv(data.camera(), 50, 150) > 0).sum())
