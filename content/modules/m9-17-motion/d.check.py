import numpy as np
import cv2
import torch
from skimage import data
from labcheck import close, need, ensure


def textured(I0, thr=20):
    Iy, Ix = np.gradient(I0)
    from scipy import ndimage
    f = lambda a: ndimage.uniform_filter(a, 7, mode="nearest")
    a, b, c = f(Ix * Ix), f(Ix * Iy), f(Iy * Iy)
    m = (a + c) / 2 - np.sqrt(((a - c) / 2) ** 2 + b * b) > thr
    m[:40] = m[-40:] = False; m[:, :40] = m[:, -40:] = False
    return m


def check(ns, R):
    I0n = cv2.GaussianBlur(data.camera().astype(np.float64), (0, 0), 1.0)
    I0 = torch.tensor(I0n)
    m = textured(I0n)

    def wp():
        f = torch.zeros(2, 512, 512, dtype=torch.float64); f[0] = 3; f[1] = -2
        close(need(ns, "warp")(I0, f)[50:-50, 50:-50].numpy(), I0n[48:-52, 53:-47], atol=1e-9, what="tamsayı kayma")
    R.test("warp", wp)

    def pl():
        rows = []
        for u, v in [(0.4, -0.3), (3.2, -1.7), (9.5, 6.0)]:
            I1 = cv2.warpAffine(I0n, np.float64([[1, 0, u], [0, 1, v]]), (512, 512), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
            fl = need(ns, "pyr_lk")(I0, torch.tensor(I1)).numpy()
            e = np.hypot(fl[0] - u, fl[1] - v)[m]
            one = ns["lk_level"](I0, torch.tensor(I1), torch.zeros(2, 512, 512, dtype=torch.float64)).numpy()
            e1 = np.hypot(one[0] - u, one[1] - v)[m]
            rows.append([f"({u}, {v})", f"{np.median(e1):.3f}", f"{np.median(e):.3f}", f"%{(e < 0.5).mean() * 100:.0f}"])
            ensure(np.median(e) < 0.1 and (e < 0.5).mean() > 0.95, f"({u}, {v}): medyan hata {np.median(e):.3f}")
        R.table("Yoğun LK, dokulu pikseller", ["gerçek akış", "tek seviye medyan hata", "4 seviye medyan hata", "< 0.5 px"], rows)
    R.test("pyr_lk büyük hareketi yakalar", pl)

    def fb():
        I1 = torch.tensor(cv2.warpAffine(I0n, np.float64([[1, 0, 4.3], [0, 1, 2.1]]), (512, 512), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT))
        f01, f10 = ns["pyr_lk"](I0, I1), ns["pyr_lk"](I1, I0)
        e = need(ns, "fb_error")(f01, f10).numpy()[m]
        ensure((e < 0.5).mean() > 0.9, f"tutarlı piksel oranı {(e < 0.5).mean():.2f}")
        return f"ileri–geri hata < 0.5 px: %{(e < 0.5).mean() * 100:.0f} (dokulu pikseller)"
    R.test("fb_error", fb)

    def fr():
        bg = I0n * 0.6
        obj = cv2.GaussianBlur(data.coins()[100:180, 100:180].astype(np.float64), (0, 0), 1.0)
        frames = []
        for t in range(3):
            f = bg.copy(); x0 = 150 + 6 * t; y0 = 200 + 3 * t
            f[y0:y0 + 80, x0:x0 + 80] = obj
            frames.append(f)
        A, B, C = (torch.tensor(f) for f in frames)
        fl = ns["pyr_lk"](A, C)
        mc = need(ns, "fruc_mc")(A, C, fl).numpy()
        av = (frames[0] + frames[2]) / 2
        reg = (slice(190, 300), slice(140, 260))
        P = lambda x: 10 * np.log10(255 ** 2 / ((x[reg] - frames[1][reg]) ** 2).mean())
        ensure(P(mc) > P(av) + 6, f"MC {P(mc):.1f} dB, ortalama {P(av):.1f} dB")
        R.table("Ara kare (nesne 12 px kayıyor), hareketli bölge", ["yöntem", "PSNR (dB)"], [["kare tekrarı", f"{P(frames[0]):.1f}"], ["ortalama (hayalet)", f"{P(av):.1f}"], ["LK + hareket telafisi", f"{P(mc):.1f}"]])
    R.test("fruc_mc", fr)
