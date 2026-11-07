import numpy as np
import cv2
from skimage import data
from labcheck import close, need, ensure


def check(ns, R):
    ops = need(ns, "operators")()
    f = data.camera().astype(np.float32)

    def kernels():
        close(cv2.filter2D(f, -1, ops["sobel"][0]), cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3), atol=1e-2, what="Sobel x")
        close(cv2.filter2D(f, -1, ops["sobel"][1]), cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3), atol=1e-2, what="Sobel y")
        close(cv2.filter2D(f, -1, ops["scharr"][0]), cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=-1), atol=1e-2, what="Scharr x")
        ensure(ops["roberts"][0].shape == (2, 2) and ops["prewitt"][0].sum() == 0, "Roberts 2×2, Prewitt sıfır toplamlı")
    R.test("çekirdekler OpenCV ile tutarlı", kernels)

    def orient():
        angles = [5, 20, 35, 50, 65, 80]
        rows = []
        res = {}
        for n in ("prewitt", "sobel", "scharr"):
            e = need(ns, "orientation_error")(n, angles)
            res[n] = float(np.max(e))
            rows.append([n, f"{np.mean(e):.2f}°", f"{np.max(e):.2f}°"])
        ensure(res["scharr"] <= res["sobel"] <= res["prewitt"] + 1e-6, f"Scharr ≤ Sobel ≤ Prewitt bekleniyordu: {res}")
        R.table("Sinüzoidal desende yön hatası (derece)", ["operatör", "ortalama", "en büyük"], rows)
    R.test("yön doğruluğu: Scharr en iyi", orient)
