import numpy as np
from skimage import data
from labcheck import close, no_loops, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("döngü yok", lambda: no_loops(src))
    img = data.camera()

    def bp():
        P = need(ns, "bit_planes")(img)
        ensure(P.shape == (8,) + img.shape and set(np.unique(P)) <= {0, 1}, "(8, H, W) ve yalnız 0/1")
        rec = (P.astype(np.int64) * (2 ** np.arange(8))[:, None, None]).sum(0)
        ensure(np.array_equal(rec, img), "Σ 2^k · düzlem_k = görüntü")
        ensure(np.array_equal(P[7], (img >= 128).astype(np.uint8)), "düzlem 7 = (img ≥ 128)")
    R.test("bit_planes: yeniden kurulum", bp)

    def rq():
        for b in (1, 3, 5, 8):
            q = need(ns, "requantize")(img, b)
            ensure(len(np.unique(q)) <= 2 ** b and q.dtype == np.uint8, f"{b} bit: {len(np.unique(q))} seviye")
            ensure(np.all(q <= img) and np.all(img - q < 2 ** (8 - b)), f"{b} bit: aşağı yuvarlama")
        ensure(np.array_equal(ns["requantize"](img, 8), img), "8 bit özdeşlik")
    R.test("requantize", rq)
