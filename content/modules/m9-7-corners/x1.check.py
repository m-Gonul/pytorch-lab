import numpy as np
import cv2
from skimage import data
from labcheck import close, need, ensure


def check(ns, R):
    img = data.checkerboard() / 255.0

    def eig():
        l1, l2 = need(ns, "eigen_maps")(img)
        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3) / 8; gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3) / 8
        b = lambda a: cv2.GaussianBlur(a, (0, 0), 1.5)
        a, bb, c = b(gx * gx), b(gx * gy), b(gy * gy)
        ensure(np.all(l1 >= l2 - 1e-12), "λ1 ≥ λ2")
        close(l1 + l2, a + c, atol=1e-10, what="iz")
        close(l1 * l2, a * c - bb * bb, atol=1e-10, what="determinant")
        close(need(ns, "harris_from_eigs")(l1, l2, 0.04), a * c - bb * bb - 0.04 * (a + c) ** 2, atol=1e-10, what="R")
        flat = (l1 < 1e-4).mean(); edge = ((l1 > 1e-3) & (l2 < 1e-4)).mean(); corner = (l2 > 1e-3).mean()
        return f"düz %{flat * 100:.0f}, kenar %{edge * 100:.0f}, köşe %{corner * 100:.1f}"
    R.test("özdeğerler, iz ve determinant", eig)

    def sweep():
        rows = []
        counts = {}
        for var in (0.006, 0.01):
            noisy = img + np.random.default_rng(0).normal(0, np.sqrt(var), img.shape)
            l1, l2 = ns["eigen_maps"](noisy)
            for k in (0.04, 0.1, 0.249):
                Rm = ns["harris_from_eigs"](l1, l2, k)
                for T in (1e-4, 1e-3):
                    counts[var, k, T] = need(ns, "count_corners")(Rm, T)
                rows.append([var, k, counts[var, k, 1e-4], counts[var, k, 1e-3]])
        for var in (0.006, 0.01):
            ensure(counts[var, 0.04, 1e-4] >= counts[var, 0.1, 1e-4] >= counts[var, 0.249, 1e-4], "k büyüdükçe köşe sayısı azalmalı")
            ensure(counts[var, 0.04, 1e-4] >= counts[var, 0.04, 1e-3], "T büyüdükçe köşe sayısı azalmalı")
        R.table("Gürültülü damalı tahta (gerçek iç köşe: 49)", ["gürültü σ²", "k", "T = 1e-4", "T = 1e-3"], rows,
                note="Düşük T ve k'da gürültü sahte köşeler üretir; k → 0.25'te gerçek köşeler bile kaybolmaya başlar.")
    R.test("k ve T taraması", sweep)
