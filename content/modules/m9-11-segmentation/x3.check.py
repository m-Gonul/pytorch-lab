import numpy as np
from scipy import ndimage
from skimage import data
from labcheck import need, ensure


def check(ns, R):
    img = data.coins()

    def rg():
        seeds = [(20, 20), (110, 50), (110, 100), (20, 25)]
        got = need(ns, "region_grow")(img, seeds, 25)
        want = np.zeros(img.shape, np.int32)
        for n, (y, x) in enumerate(seeds, 1):
            if want[y, x]:
                continue
            ok = (np.abs(img.astype(int) - int(img[y, x])) <= 25) & (want == 0)
            lab, _ = ndimage.label(ok, np.ones((3, 3)))
            want[lab == lab[y, x]] = n
        ensure(np.array_equal(got, want), "başvuru ile farklı")
        return f"bölge boyutları: {np.bincount(got.ravel())[1:].tolist()} (4. tohum 1. bölgenin içinde, atlanır)"
    R.test("region_grow", rg)
