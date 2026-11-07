import numpy as np
from skimage import data
from labcheck import no_loops, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("döngü yok", lambda: no_loops(src))
    img = data.camera()

    def sl():
        levels = [50, 100, 101, 200]
        cols = [(i * 50, 255 - i * 50, i * 20) for i in range(5)]
        got = need(ns, "intensity_slice")(img, levels, cols)
        want = np.zeros(img.shape + (3,), np.uint8)
        for k in range(5):
            lo = -1 if k == 0 else levels[k - 1]; hi = 256 if k == 4 else levels[k]
            want[(img >= max(lo, 0)) & (img < hi)] = cols[k]
        ensure(got.shape == want.shape and got.dtype == np.uint8 and np.array_equal(got, want), "dilimleme")
        ensure(np.array_equal(ns["intensity_slice"](np.array([[99, 100, 101]], np.uint8), [100], [(0, 0, 0), (9, 9, 9)])[0, :, 0], [0, 9, 9]), "eşik kendisi üst dilime girer")
    R.test("intensity_slice", sl)

    def sp():
        got = need(ns, "sin_pseudocolor")(img, 1.0, (0.0, 2.0, 4.0))
        f = img.astype(np.float64)
        for c, p in enumerate((0.0, 2.0, 4.0)):
            ensure(np.array_equal(got[..., c], np.round(255 * (1 + np.sin(2 * np.pi * f / 255 + p)) / 2).astype(np.uint8)), f"kanal {c}")
    R.test("sin_pseudocolor", sp)
