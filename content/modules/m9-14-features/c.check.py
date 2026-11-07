import numpy as np
import cv2
from skimage import data
from skimage.feature import hog as skhog
from labcheck import close, no_loops, need, ensure


def kernels(L):
    l = L // 3; r = L // 2; wl = 2 * l - 1
    Kyy = np.zeros((L, L)); c = r
    Kyy[:, c - wl // 2:c + wl // 2 + 1] = 1; Kyy[l:2 * l, c - wl // 2:c + wl // 2 + 1] = -2
    Kxx = Kyy.T.copy()
    Kxy = np.zeros((L, L))
    Kxy[c - l:c, c - l:c] = 1; Kxy[c + 1:c + 1 + l, c + 1:c + 1 + l] = 1
    Kxy[c - l:c, c + 1:c + 1 + l] = -1; Kxy[c + 1:c + 1 + l, c - l:c] = -1
    return Kxx, Kyy, Kxy


def check(ns, R):
    src = ns["__source__"]
    R.test("döngü yok", lambda: no_loops(src))
    for name in ("camera", "coins", "moon"):
        def t(name=name):
            im = getattr(data, name)()[:200, :150]
            for cell, blk, nb in [(8, 2, 9), (6, 3, 12)]:
                want = skhog(im, orientations=nb, pixels_per_cell=(cell, cell), cells_per_block=(blk, blk), block_norm="L2-Hys", feature_vector=True)
                close(need(ns, "hog")(im, nb, cell, blk), want, atol=1e-6, what=f"hücre {cell}, blok {blk}, {nb} yön")
        R.test(f"hog = skimage ({name})", t)

    def fh():
        im = data.camera()[100:260, 150:330].astype(np.float64)
        rows = []
        for L in (9, 15, 27):
            Kxx, Kyy, Kxy = kernels(L)
            f = lambda K: cv2.filter2D(im, -1, K, borderType=cv2.BORDER_CONSTANT) / L ** 2
            want = f(Kxx) * f(Kyy) - (0.9 * f(Kxy)) ** 2
            close(need(ns, "fast_hessian")(im, L), want, atol=1e-6, what=f"L = {L}")
            rows.append([L, f"{1.2 * L / 9:.1f}", f"{np.abs(Kxx).sum():.0f}"])
        R.table("SURF kutu süzgeçleri", ["L", "σ karşılığı", "Dxx ağırlıklarının mutlak toplamı"], rows,
                note="Integral görüntüyle her kutu 4 erişim: Dxx ve Dyy 2 kutu (8 erişim), Dxy 4 kutu (16 erişim), L'den bağımsız.")
    R.test("fast_hessian = açık kutu çekirdekleri", fh)
