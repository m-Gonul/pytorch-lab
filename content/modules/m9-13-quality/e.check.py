import numpy as np
import torch
from scipy import ndimage
from skimage import data, color
from labcheck import close, max_loops, need, ensure, bench, fmt_time

W = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)


def ref_ms(x, y):
    f = lambda a: ndimage.gaussian_filter(a, 1.5, truncate=3.5, mode="reflect")
    C1, C2 = 6.5025, 58.5225
    out = 1.0
    for j, w in enumerate(W):
        ux, uy = f(x), f(y)
        vx, vy, vxy = f(x * x) - ux * ux, f(y * y) - uy * uy, f(x * y) - ux * uy
        cs = (2 * vxy + C2) / (vx + vy + C2)
        if j < 4:
            out *= max(cs[5:-5, 5:-5].mean(), 0) ** w
            H, Wd = x.shape[0] // 2 * 2, x.shape[1] // 2 * 2
            x = x[:H, :Wd].reshape(H // 2, 2, Wd // 2, 2).mean((1, 3)); y = y[:H, :Wd].reshape(H // 2, 2, Wd // 2, 2).mean((1, 3))
        else:
            l = (2 * ux * uy + C1) / (ux * ux + uy * uy + C1)
            out *= max((l * cs)[5:-5, 5:-5].mean(), 0) ** w
    return out


def check(ns, R):
    src = ns["__source__"]
    R.test("yalnız ölçek döngüsü", lambda: max_loops(src, 1, func="ms_ssim"))
    rng = np.random.default_rng(0)
    ref = np.stack([data.camera()[:256, :256], color.rgb2gray(data.astronaut())[:256, :256] * 255, data.moon()[:256, :256], data.coins()[:256, :256]]).astype(np.float64)
    dist = np.clip(ref + rng.normal(0, 25, ref.shape), 0, 255)

    def eq():
        got = need(ns, "ms_ssim")(torch.tensor(ref), torch.tensor(dist)).numpy()
        want = np.array([ref_ms(a, b) for a, b in zip(ref, dist)])
        close(got, want, atol=1e-10, what="görüntü başına başvuru")
        close(ns["ms_ssim"](torch.tensor(ref), torch.tensor(ref)).numpy(), np.ones(4), atol=1e-12, what="özdeş → 1")
    R.test("ms_ssim = başvuru (4 görüntülük yığın)", eq)
    if "ms_ssim" not in ns:
        return

    def tab():
        img = ref[0]
        ss = lambda a: ns["ms_ssim"](torch.tensor(img[None]), torch.tensor(a[None])).item()
        import cv2
        rows = [["Gauss gürültüsü σ = 25", f"{ss(np.clip(img + rng.normal(0, 25, img.shape), 0, 255)):.3f}"],
                ["Gauss bulanıklık σ = 2", f"{ss(cv2.GaussianBlur(img, (0, 0), 2)):.3f}"],
                ["ortalama +15", f"{ss(img + 15):.3f}"]]
        R.table("camera, MS-SSIM", ["bozulma", "MS-SSIM"], rows)
    R.test("MS-SSIM tablosu", tab)
    big_r, big_d = torch.tensor(np.tile(ref, (4, 1, 1))), torch.tensor(np.tile(dist, (4, 1, 1)))
    t1 = bench(lambda: [ref_ms(a, b) for a, b in zip(big_r.numpy(), big_d.numpy())], repeat=1, number=1)
    t2 = bench(lambda: ns["ms_ssim"](big_r, big_d), repeat=3)
    R.table("16 × 256²", ["yöntem", "süre"], [["görüntü döngüsü (scipy)", fmt_time(t1)], ["toplu tensör", fmt_time(t2)]])
