import numpy as np
import cv2
from skimage import data
from labcheck import close, no_imports, need, ensure


def sp(img, p, seed):
    rng = np.random.default_rng(seed)
    out = img.copy()
    u = rng.random(img.shape)
    out[u < p / 2] = 0
    out[(u >= p / 2) & (u < p)] = 255
    return out


def ref_adaptive(img, smax=7):
    H, W = img.shape
    P = np.pad(img, smax // 2, mode="edge").astype(int)
    out = np.zeros_like(img)
    for y in range(H):
        for x in range(W):
            k = 3
            while True:
                r = k // 2
                win = P[y + smax // 2 - r:y + smax // 2 + r + 1, x + smax // 2 - r:x + smax // 2 + r + 1]
                zmin, zmax, zmed = win.min(), win.max(), int(np.median(win))
                if zmin < zmed < zmax:
                    z = int(img[y, x])
                    out[y, x] = z if zmin < z < zmax else zmed
                    break
                k += 2
                if k > smax:
                    out[y, x] = zmed
                    break
    return out


def check(ns, R):
    src = ns["__source__"]
    R.test("cv2/numpy kullanılmıyor", lambda: no_imports(src, {"cv2", "numpy", "skimage", "scipy"}))
    crop = sp(data.camera()[200:248, 220:268], 0.2, 0)

    def med():
        for k in (3, 5):
            got = np.array(need(ns, "median_filter")(crop.tolist(), k), np.uint8)
            ensure(np.array_equal(got, cv2.medianBlur(crop, k)), f"{k}×{k}: cv2.medianBlur ile farklı")
    R.test("median_filter = cv2.medianBlur (birebir)", med)

    def cw():
        got = np.array(need(ns, "cw_median")(crop.tolist(), 3, 3), np.uint8)
        P = np.pad(crop, 1, mode="edge")
        win = np.lib.stride_tricks.sliding_window_view(P, (3, 3)).reshape(48, 48, 9)
        want = np.median(np.concatenate([win, np.repeat(crop[..., None], 2, -1)], -1), -1).astype(np.uint8)
        ensure(np.array_equal(got, want), "merkez ağırlıklı median")
        ensure(np.array_equal(np.array(ns["cw_median"](crop.tolist(), 5, 1), np.uint8), cv2.medianBlur(crop, 5)), "w = 1 → normal median")
    R.test("cw_median", cw)

    def am():
        dense = sp(data.camera()[260:300, 100:140], 0.5, 1)
        got = np.array(need(ns, "adaptive_median")(dense.tolist()), np.uint8)
        ensure(np.array_equal(got, ref_adaptive(dense)), "başvuru ile farklı")
        clean = data.camera()[260:300, 100:140]
        e_am = np.abs(got.astype(int) - clean).mean(); e_m = np.abs(cv2.medianBlur(dense, 3).astype(int) - clean).mean()
        return f"%50 tuz-biber: ortalama mutlak hata uyarlamalı {e_am:.1f}, median 3×3 {e_m:.1f}"
    R.test("adaptive_median (yoğun gürültü)", am)
