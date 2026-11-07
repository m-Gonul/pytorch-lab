import numpy as np
import cv2
from skimage import data, feature
from skimage.transform import hough_line
from labcheck import close, need, ensure


def check(ns, R):
    th = np.linspace(-np.pi / 2, np.pi / 2, 180, endpoint=False)

    def same():
        e = feature.canny(data.camera()[::2, ::2].astype(float), 2)
        acc, d = need(ns, "hough_lines")(e, th)
        A, _, D = hough_line(e, th)
        ensure(acc.shape == A.shape and np.array_equal(acc, A), "birikim dizisi skimage ile farklı")
        close(d, D, what="dists")
    R.test("hough_lines = skimage.transform.hough_line", same)

    def find():
        img = np.zeros((200, 240), np.uint8)
        truth = [(60.0, np.deg2rad(-30)), (100.0, np.deg2rad(45)), (120.0, np.deg2rad(80))]
        for rho, t in truth:
            c, s = np.cos(t), np.sin(t)
            x0, y0 = rho * c, rho * s
            p1 = (int(round(x0 - 1000 * s)), int(round(y0 + 1000 * c))); p2 = (int(round(x0 + 1000 * s)), int(round(y0 - 1000 * c)))
            cv2.line(img, p1, p2, 255, 1)
        rng = np.random.default_rng(0)
        img[rng.random(img.shape) < 0.01] = 255
        acc, d = ns["hough_lines"](img > 0, th)
        got = need(ns, "top_lines")(acc, th, d, 3)
        rows = []
        for rho, t in truth:
            best = min(got, key=lambda g: abs(g[0] - rho) + 50 * abs(g[1] - t))
            ensure(abs(best[0] - rho) <= 2 and abs(np.rad2deg(best[1] - t)) <= 2, f"(ρ = {rho}, θ = {np.rad2deg(t):.0f}°) bulunamadı")
            rows.append([f"{rho:.0f}, {np.rad2deg(t):.0f}°", f"{best[0]:.0f}, {np.rad2deg(best[1]):.0f}°"])
        R.table("%1 tuz gürültüsü altında üç doğru", ["gerçek (ρ, θ)", "bulunan"], rows)
    R.test("top_lines doğruları buluyor", find)
