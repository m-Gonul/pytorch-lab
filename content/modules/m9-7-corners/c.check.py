import numpy as np
import cv2
from skimage import data
from labcheck import close, no_loops, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("döngü yok", lambda: no_loops(src))
    img = data.camera()

    for block, ksize in [(2, 3), (3, 3), (5, 5), (3, 1)]:
        def t(block=block, ksize=ksize):
            got = need(ns, "corner_harris_cv")(img, block, ksize, 0.04)
            want = cv2.cornerHarris(img, block, ksize, 0.04)
            close(got, want, atol=1e-6 * float(np.abs(want).max()), what="cornerHarris")
            got2 = need(ns, "corner_min_eig_cv")(img, block, ksize)
            want2 = cv2.cornerMinEigenVal(img, block, ksize=ksize)
            close(got2, want2, atol=1e-6 * float(np.abs(want2).max()), what="cornerMinEigenVal")
        R.test(f"= cv2.cornerHarris / cornerMinEigenVal (block {block}, ksize {ksize})", t)

    def pts():
        rows = []
        for k, T in [(0.04, 0.01), (0.1, 0.01), (0.04, 0.1), (0.2, 0.01)]:
            p = need(ns, "harris_points")(img, k, T)
            ensure(p.ndim == 2 and p.shape[1] == 2, "(N, 2)")
            Rm = cv2.cornerHarris(img, 3, 3, k)
            vals = Rm[p[:, 1], p[:, 0]]
            ensure(np.all(np.diff(vals) <= 0), "yanıta göre azalan sırada değil")
            rows.append([k, T, len(p)])
        ensure(rows[1][2] < rows[0][2] and rows[2][2] < rows[0][2], "k ya da T büyüyünce köşe sayısı azalmalı")
        R.table("camera, 7 × 7 yerel en büyük", ["k", "T (max'ın oranı)", "köşe"], rows)
    R.test("harris_points", pts)
