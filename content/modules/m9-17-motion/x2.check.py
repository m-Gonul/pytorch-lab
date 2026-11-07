import numpy as np
import cv2
from skimage import data
from labcheck import need, ensure


def check(ns, R):
    I0 = cv2.GaussianBlur(data.camera()[128:384, 128:384].astype(np.float64), (0, 0), 1.5)

    def t():
        H, W = I0.shape
        M = cv2.getRotationMatrix2D((W / 2, H / 2), 1.0, 1.0)
        I1 = cv2.warpAffine(I0, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        ys, xs = np.mgrid[:H, :W].astype(float)
        p = np.stack([xs, ys, np.ones_like(xs)], -1) @ M.T
        gu, gv = p[..., 0] - xs, p[..., 1] - ys
        u, v = need(ns, "horn_schunck")(I0, I1, 10.0, 300)
        inner = (slice(30, -30), slice(30, -30))
        e = np.hypot(u - gu, v - gv)[inner]
        Iy, Ix = np.gradient(I0)
        from scipy import ndimage
        f = lambda a: ndimage.uniform_filter(a, 7, mode="nearest")
        a, b, c = f(Ix * Ix), f(Ix * Iy), f(Iy * Iy); p_, q_ = f(Ix * (I1 - I0)), f(Iy * (I1 - I0))
        det = a * c - b * b
        lk_u, lk_v = np.where(det > 1e-3, (-c * p_ + b * q_) / np.where(det > 1e-3, det, 1), 0), np.where(det > 1e-3, (b * p_ - a * q_) / np.where(det > 1e-3, det, 1), 0)
        el = np.hypot(lk_u - gu, lk_v - gv)[inner]
        ensure(np.mean(e) < np.mean(el), f"HS ortalama hata {np.mean(e):.3f}, LK {np.mean(el):.3f}")
        R.table("1° dönme (akış merkezden uzaklıkla büyür, en çok ≈ 2.2 px)", ["yöntem", "ortalama EPE", "%90'lık EPE"],
                [["LK (7 × 7, tek adım)", f"{np.mean(el):.3f}", f"{np.percentile(el, 90):.3f}"], ["Horn–Schunck (α = 10, 300 it.)", f"{np.mean(e):.3f}", f"{np.percentile(e, 90):.3f}"]],
                note="LK düz bölgelerde akış veremez (0 kalır); HS düzgünlük terimiyle akışı dokusuz bölgelere yayar.")
    R.test("Horn–Schunck vs LK", t)
