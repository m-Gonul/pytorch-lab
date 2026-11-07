import numpy as np
from scipy import ndimage
from skimage import data
from labcheck import close, no_loops, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("döngü yok", lambda: no_loops(src))
    img = data.camera()[200:260, 200:260]
    f = img.astype(np.float64) + 1
    R.test("geometric_mean", lambda: close(need(ns, "geometric_mean")(img, 3), ndimage.generic_filter(f, lambda w: np.exp(np.log(w).mean()), 3, mode="nearest") - 1, atol=1e-8))
    R.test("harmonic_mean", lambda: close(need(ns, "harmonic_mean")(img, 3), ndimage.generic_filter(f, lambda w: 9 / (1 / w).sum(), 3, mode="nearest") - 1, atol=1e-8))
    R.test("contraharmonic Q = 1.5", lambda: close(need(ns, "contraharmonic")(img, 3, 1.5), ndimage.generic_filter(f, lambda w: (w ** 2.5).sum() / (w ** 1.5).sum(), 3, mode="nearest") - 1, atol=1e-6))

    def sign():
        clean = data.camera()
        rng = np.random.default_rng(0)
        pepper = clean.copy(); pepper[rng.random(clean.shape) < 0.1] = 0
        salt = clean.copy(); salt[rng.random(clean.shape) < 0.1] = 255
        e = lambda a: np.abs(a - clean).mean()
        rows = [["biber", f"{e(ns['contraharmonic'](pepper, 3, 1.5)):.1f}", f"{e(ns['contraharmonic'](pepper, 3, -1.5)):.1f}"],
                ["tuz", f"{e(ns['contraharmonic'](salt, 3, 1.5)):.1f}", f"{e(ns['contraharmonic'](salt, 3, -1.5)):.1f}"]]
        ensure(e(ns["contraharmonic"](pepper, 3, 1.5)) < e(ns["contraharmonic"](pepper, 3, -1.5)), "biberde Q > 0 daha iyi")
        ensure(e(ns["contraharmonic"](salt, 3, -1.5)) < e(ns["contraharmonic"](salt, 3, 1.5)), "tuzda Q < 0 daha iyi")
        R.table("Ortalama mutlak hata", ["gürültü", "Q = +1.5", "Q = −1.5"], rows)
    R.test("Q'nun işareti: biber için +, tuz için −", sign)
