import numpy as np
from scipy import ndimage
from skimage import data
from labcheck import need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("hazır fonksiyon yok", lambda: ensure("binary_fill_holes" not in src and "label(" not in src, "binary_fill_holes / label kullanma"))
    A = data.coins()[30:180, 20:200] > 110

    def fh():
        got = need(ns, "fill_holes")(A)
        ensure(np.array_equal(got, ndimage.binary_fill_holes(A)), "scipy binary_fill_holes ile farklı")
        return f"{int(got.sum() - A.sum())} delik pikseli dolduruldu"
    R.test("fill_holes", fh)

    def cc():
        B = ndimage.binary_opening(A, np.ones((3, 3)))
        lab = need(ns, "components")(B)
        ref, n = ndimage.label(B, np.ones((3, 3)))
        ensure(lab.max() == n, f"{lab.max()} bileşen, {n} bekleniyordu")
        firsts = [np.argmax((ref == i).ravel()) for i in range(1, n + 1)]
        order = np.argsort(firsts)
        for new, old in enumerate(order, 1):
            ensure(np.array_equal(lab == new, ref == old + 1), f"bileşen {new}")
        return f"{n} bileşen"
    R.test("components (8-bağlantı, satır sırası)", cc)
