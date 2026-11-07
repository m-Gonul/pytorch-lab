import numpy as np
from scipy import ndimage
from skimage import data
from labcheck import close, no_loops, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("döngü yok", lambda: no_loops(src))
    img = data.camera()

    def t():
        f = img.astype(np.float64)
        m = ndimage.uniform_filter(f, 3, mode="mirror")
        s = np.sqrt(np.maximum(ndimage.uniform_filter(f * f, 3, mode="mirror") - m * m, 0))
        sel = (m <= 0.4 * f.mean()) & (s >= 0.02 * f.std()) & (s <= 0.4 * f.std())
        want = np.clip(np.where(sel, 4 * f, f), 0, 255).astype(np.uint8)
        got = need(ns, "local_enhance")(img)
        diff = np.abs(got.astype(int) - want.astype(int))
        ensure(got.dtype == np.uint8 and (diff > 1).mean() < 1e-3, f"{(diff > 1).mean():.2%} piksel farklı")
        return f"seçilen piksel oranı {sel.mean():.1%}"
    R.test("local_enhance (Gonzalez yerel istatistik kuralı)", t)
