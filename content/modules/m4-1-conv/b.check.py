import numpy as np
from labcheck import close, no_imports, need, ensure


def ref(x, w, b, s, p):
    x = np.pad(np.asarray(x, float), ((0, 0), (p, p), (p, p))); w = np.asarray(w, float)
    F, C, KH, KW = w.shape
    OH, OW = (x.shape[1] - KH) // s + 1, (x.shape[2] - KW) // s + 1
    return np.array([[[(x[:, i * s:i * s + KH, j * s:j * s + KW] * w[f]).sum() + b[f] for j in range(OW)] for i in range(OH)] for f in range(F)])


def check(ns, R):
    src = ns["__source__"]
    R.test("numpy/torch kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))
    R.test("elle örnek", lambda: close(need(ns, "conv2d_loops")([[[1, 2, 0], [0, 1, 3], [2, 1, 0]]], [[[[1, 0], [-1, 2]]]], [0]), [[[3, 7], [0, 0]]]))
    R.test("pad2d", lambda: close(need(ns, "pad2d")([[[1, 2]]], 1), [[[0, 0, 0, 0], [0, 1, 2, 0], [0, 0, 0, 0]]]))
    rng = np.random.default_rng(0)

    def rnd():
        x = rng.normal(size=(2, 7, 6)); w = rng.normal(size=(3, 2, 3, 3)); b = rng.normal(size=3)
        for s, p in [(1, 0), (2, 1), (3, 2)]:
            close(need(ns, "conv2d_loops")(x.tolist(), w.tolist(), b.tolist(), s, p), ref(x, w, b, s, p), what=f"stride {s}, pad {p}")
    R.test("2 kanal → 3 çekirdek, farklı stride/padding", rnd)
