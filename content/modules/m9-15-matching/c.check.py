import numpy as np
import cv2
from skimage import data
from labcheck import close, no_loops, need, ensure


def corners_err(H, Ht, size=500):
    c = np.array([[[0, 0], [size, 0], [size, size], [0, size]]], np.float64)
    return float(np.abs(cv2.perspectiveTransform(c, H) - cv2.perspectiveTransform(c, Ht)).max())


def check(ns, R):
    src_ = ns["__source__"]
    R.test("döngü yok", lambda: no_loops(src_))
    img = data.camera().astype(np.float32)

    def nm():
        for (y, x, h, w) in [(200, 250, 40, 50), (10, 10, 21, 33), (400, 300, 64, 64)]:
            t = img[y:y + h, x:x + w].copy()
            got, want = need(ns, "ncc_map")(img, t), cv2.matchTemplate(img, t, cv2.TM_CCOEFF_NORMED)
            ensure(got.shape == want.shape, "şekil")
            tz = t.astype(np.float64) - t.mean()
            rr = np.random.default_rng(h)
            for yy, xx in zip(rr.integers(0, got.shape[0], 200), rr.integers(0, got.shape[1], 200)):
                wz = img[yy:yy + h, xx:xx + w].astype(np.float64); wz = wz - wz.mean()
                den = np.sqrt((tz ** 2).sum() * (wz ** 2).sum())
                if den > 1e-3:
                    close(got[yy, xx], (tz * wz).sum() / den, atol=1e-9, what=f"({yy}, {xx}) doğrudan tanım")
            f64 = img.astype(np.float64)
            I1 = np.pad(f64, ((1, 0), (1, 0))).cumsum(0).cumsum(1); I2 = np.pad(f64 ** 2, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
            Sb = lambda I: I[h:, w:] - I[:-h, w:] - I[h:, :-w] + I[:-h, :-w]
            tex = np.sqrt(np.maximum(Sb(I2) / (h * w) - (Sb(I1) / (h * w)) ** 2, 0)) > 5
            close(got[tex], want[tex], atol=5e-3, what=f"şablon {h} × {w}, dokulu pencerelerde cv2")
            ensure(np.unravel_index(np.argmax(got), got.shape) == (y, x), "en iyi konum")
    R.test("ncc_map = cv2.matchTemplate", nm)

    rng = np.random.default_rng(0)
    Ht = np.array([[0.9, 0.1, 30], [-0.05, 1.1, -20], [1e-4, 2e-4, 1]])

    def d():
        src = rng.uniform(0, 500, (30, 2))
        dst = cv2.perspectiveTransform(src[None], Ht)[0]
        q, T = need(ns, "normalize_points")(src)
        close(q.mean(0), [0, 0], atol=1e-9, what="ağırlık merkezi"); close(np.linalg.norm(q, axis=1).mean(), np.sqrt(2), what="ortalama uzaklık")
        close(need(ns, "dlt")(src, dst), Ht, atol=1e-8, what="gürültüsüz DLT")
        noisy = dst + rng.normal(0, 0.5, dst.shape)
        Hc, _ = cv2.findHomography(src, noisy, 0)
        H = ns["dlt"](src, noisy)
        ensure(corners_err(H, Ht) < corners_err(Hc, Ht) + 1.0, "gürültülü DLT OpenCV ile benzer doğrulukta olmalı")
    R.test("normalize_points ve dlt", d)

    def r():
        rows = []
        for ratio in (0.2, 0.4, 0.6):
            src = rng.uniform(0, 500, (200, 2))
            dst = cv2.perspectiveTransform(src[None], Ht)[0] + rng.normal(0, 0.5, (200, 2))
            out = rng.random(200) < ratio
            dst[out] = rng.uniform(0, 500, (out.sum(), 2))
            H, m = need(ns, "ransac")(src, dst, 3.0, 1000)
            Hc, mc = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
            e, ec = corners_err(H, Ht), corners_err(Hc, Ht)
            ensure(e < 3 and (m == ~out).mean() > 0.95, f"%{int(ratio * 100)} aykırı: köşe hatası {e:.2f}, maske doğruluğu {(m == ~out).mean():.2f}")
            rows.append([f"%{int(ratio * 100)}", int(m.sum()), int((~out).sum()), f"{e:.2f}", f"{ec:.2f}"])
        R.table("200 eş, σ = 0.5 gürültü", ["aykırı", "bulunan içerdeğer", "gerçek içerdeğer", "köşe hatası (px)", "cv2 köşe hatası"], rows)
    R.test("ransac (tüm hipotezler tek SVD'de)", r)
