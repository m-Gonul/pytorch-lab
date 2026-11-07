import math
import numpy as np
import torch
from scipy import ndimage
from skimage import data
from labcheck import close, need, ensure


def ref_harris(img, sd, si, k):
    def g(s):
        r = int(math.ceil(3 * s)); t = np.arange(-r, r + 1); w = np.exp(-t ** 2 / (2 * s * s)); return w / w.sum()
    sep = lambda a, w: ndimage.correlate1d(ndimage.correlate1d(a, w, 1, mode="mirror"), w, 0, mode="mirror")
    x = sep(img, g(sd))
    gx = ndimage.correlate1d(x, [-0.5, 0, 0.5], 1, mode="nearest"); gy = ndimage.correlate1d(x, [-0.5, 0, 0.5], 0, mode="nearest")
    w = g(si)
    a, b, c = sep(gx * gx, w), sep(gx * gy, w), sep(gy * gy, w)
    return a * c - b * b - k * (a + c) ** 2


def check(ns, R):
    img = torch.tensor(data.camera(), dtype=torch.float32) / 255

    def h():
        crop = img[100:200, 150:280]
        for sd, si, k in [(1.0, 2.0, 0.04), (1.5, 3.0, 0.1)]:
            got = need(ns, "harris_torch")(crop, sd, si, k).double().numpy()
            want = ref_harris(crop.double().numpy(), sd, si, k)
            close(got, want, atol=1e-5 * np.abs(want).max(), what=f"σ_d = {sd}, σ_i = {si}")
    R.test("harris_torch", h)

    def det():
        Rm = need(ns, "harris_torch")(img)
        p = need(ns, "detect")(Rm, 50, 4, 16)
        ensure(tuple(p.shape) == (50, 2), "(n, 2)")
        x, y = p[:, 0].long(), p[:, 1].long()
        ensure(bool(((x >= 16) & (x < 512 - 16) & (y >= 16) & (y < 512 - 16)).all()), "kenar bölgesinde tespit var")
        vals = Rm[y, x]
        mx = torch.nn.functional.max_pool2d(Rm[None, None], 9, 1, 4)[0, 0]
        ensure(bool((vals == mx[y, x]).all()), "yerel en büyük değil")
        ensure(bool((vals[:-1] >= vals[1:]).all()), "azalan sırada değil")
    R.test("detect", det)

    def rep():
        f = need(ns, "repeatability")
        rows = [[f"{a}°", s, f"{f(img, a, s):.2f}"] for a, s in [(0, 1.0), (15, 1.0), (30, 1.0), (45, 1.0), (0, 1.5), (0, 2.0)]]
        r30, rs2 = float(rows[2][2]), float(rows[5][2])
        ensure(float(rows[0][2]) > 0.99, "dönüşüm yokken tekrarlanabilirlik 1 olmalı")
        ensure(r30 > 0.75, f"30° dönmede {r30:.2f}: Harris dönmeye dayanıklı olmalı")
        ensure(rs2 < r30 - 0.15, f"2× ölçekte {rs2:.2f}, dönmede {r30:.2f}: ölçek daha çok zarar vermeli")
        fixed3, ad3 = f(img, 0, 3.0, False, 6.0), f(img, 0, 3.0, True, 6.0)
        ensure(ad3 > fixed3 + 0.05, f"3× ölçekte uyarlamalı {ad3:.2f}, sabit {fixed3:.2f}")
        rows.append(["0°", "3.0 (tol 6)", f"sabit σ {fixed3:.2f} / uyarlamalı σ {ad3:.2f}"])
        R.table("camera, 200 köşe, 2 piksel tolerans", ["dönme", "ölçek", "tekrarlanabilirlik"], rows,
                note="Ölçeği bilinmeyen bir eşleştirmede σ'yı nasıl seçeceğiz? Cevap ölçek uzayı: M9.14'te SIFT.")
    R.test("dönme ve ölçek altında tekrarlanabilirlik", rep)
