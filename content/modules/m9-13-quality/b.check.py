import math
import numpy as np
from skimage import data
from labcheck import close, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("numpy kullanılmıyor", lambda: no_imports(src, {"numpy", "skimage", "cv2", "scipy"}))
    rng = np.random.default_rng(0)
    x = data.camera()[100:132, 100:132].astype(float).ravel()
    y = np.clip(x + rng.normal(0, 20, x.shape), 0, 255)

    def mp():
        close(need(ns, "mse")(x.tolist(), y.tolist()), ((x - y) ** 2).mean(), what="MSE")
        close(need(ns, "psnr")(x.tolist(), y.tolist()), 10 * np.log10(255 ** 2 / ((x - y) ** 2).mean()), what="PSNR")
        ensure(ns["psnr"]([1, 2], [1, 2]) == float("inf"), "özdeş → inf")
    R.test("mse, psnr", mp)

    def ss():
        C1, C2 = 6.5025, 58.5225
        mx, my = x.mean(), y.mean(); vx, vy = x.var(ddof=1), y.var(ddof=1); c = np.cov(x, y)[0, 1]
        want = (2 * mx * my + C1) * (2 * c + C2) / ((mx ** 2 + my ** 2 + C1) * (vx + vy + C2))
        close(need(ns, "ssim_global")(x.tolist(), y.tolist()), want, atol=1e-12, what="SSIM")
        l, cc, s = need(ns, "ssim_components")(x.tolist(), y.tolist())
        close(l * cc * s, want, atol=1e-12, what="l·c·s = SSIM (C3 = C2/2)")
        close(ns["ssim_global"](x.tolist(), x.tolist()), 1.0, what="özdeş → 1")
        close(ns["ssim_global"](x.tolist(), y.tolist()), ns["ssim_global"](y.tolist(), x.tolist()), what="simetri")
        return f"l = {l:.4f}, c = {cc:.4f}, s = {s:.4f}"
    R.test("ssim_global ve bileşenleri", ss)
