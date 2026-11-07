import numpy as np
import cv2
from scipy import ndimage
from skimage import data
from labcheck import no_imports, need, ensure


def ref_nms(m, gx, gy):
    H, W = m.shape
    P = np.pad(m, 1)
    a = np.degrees(np.arctan2(gy, gx)) % 180
    sh = lambda dy, dx: P[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
    d0 = (a < 22.5) | (a >= 157.5); d1 = (a >= 22.5) & (a < 67.5); d2 = (a >= 67.5) & (a < 112.5); d3 = (a >= 112.5) & (a < 157.5)
    n1 = np.select([d0, d1, d2, d3], [sh(0, -1), sh(-1, -1), sh(-1, 0), sh(-1, 1)])
    n2 = np.select([d0, d1, d2, d3], [sh(0, 1), sh(1, 1), sh(1, 0), sh(1, -1)])
    return np.where((m < n1) | (m < n2), 0, m)


def ref_hys(g, lo, hi):
    cand = g >= lo
    lab, n = ndimage.label(cand, np.ones((3, 3)))
    ok = np.zeros(n + 1, bool); ok[np.unique(lab[g >= hi])] = True; ok[0] = False
    return ok[lab].astype(int)


def ref_zc(g, thr):
    H, W = g.shape
    out = np.zeros((H, W), int)
    c = lambda dy, dx: g[1 + dy:H - 1 + dy, 1 + dx:W - 1 + dx]
    z = np.zeros((H - 2, W - 2), bool)
    for (a, b), (cc, d) in [((0, -1), (0, 1)), ((-1, 0), (1, 0)), ((-1, -1), (1, 1)), ((-1, 1), (1, -1))]:
        p, n = c(a, b), c(cc, d)
        z |= (p * n < 0) & (np.abs(p - n) > thr)
    out[1:-1, 1:-1] = z
    return out


def check(ns, R):
    src = ns["__source__"]
    R.test("numpy/cv2 kullanılmıyor", lambda: no_imports(src, {"numpy", "cv2", "scipy", "skimage"}))
    img = cv2.GaussianBlur(data.camera()[180:230, 230:290], (5, 5), 1.2).astype(np.float64)
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0); gy = cv2.Sobel(img, cv2.CV_64F, 0, 1); m = np.hypot(gx, gy)

    def t_nms():
        got = np.array(need(ns, "nms")(m.tolist(), gx.tolist(), gy.tolist()))
        ensure(np.allclose(got, ref_nms(m, gx, gy)), "başvuru NMS ile farklı")
        return f"{(m > 0).sum()} → {(got > 0).sum()} sıfır olmayan piksel"
    R.test("nms", t_nms)

    def t_hys():
        n = ref_nms(m, gx, gy)
        lo, hi = np.percentile(n[n > 0], [50, 85])
        got = np.array(need(ns, "hysteresis")(n.tolist(), lo, hi))
        ensure(np.array_equal(got, ref_hys(n, lo, hi)), "başvuru (bağlı bileşenler) ile farklı")
        ensure(ns["hysteresis"]([[0, 40, 120, 60, 0, 70, 0]], 50, 100) == [[0, 0, 1, 1, 0, 0, 0]], "elle örnek")
    R.test("hysteresis", t_hys)

    def t_zc():
        g = cv2.Laplacian(cv2.GaussianBlur(data.camera()[100:150, 100:160].astype(np.float64), (9, 9), 1.4), cv2.CV_64F)
        for thr in (0.0, 0.04 * g.max()):
            got = np.array(need(ns, "zero_crossings")(g.tolist(), thr))
            ensure(np.array_equal(got, ref_zc(g, thr)), f"eşik {thr:.2f}")
    R.test("zero_crossings", t_zc)
