import numpy as np
from labcheck import close, no_loops, no_imports, need, ensure


def ref(x, w, b, s, p):
    x = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))
    N, C, H, W = x.shape; F, _, KH, KW = w.shape
    OH, OW = (H - KH) // s + 1, (W - KW) // s + 1
    y = np.zeros((N, F, OH, OW))
    for i in range(OH):
        for j in range(OW):
            y[:, :, i, j] = np.tensordot(x[:, :, i * s:i * s + KH, j * s:j * s + KW], w, axes=([1, 2, 3], [1, 2, 3])) + b
    return y


def check(ns, R):
    src = ns["__source__"]
    R.test("döngü yok", lambda: no_loops(src))
    R.test("torch kullanılmıyor", lambda: no_imports(src, {"torch"}))
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 3, 9, 8)); w = rng.normal(size=(4, 3, 3, 3)); b = rng.normal(size=4)

    def cv():
        for s, p in [(1, 0), (1, 1), (2, 1), (2, 2)]:
            out = need(ns, "conv2d_np")(x, w, b, s, p)
            want = ref(x, w, b, s, p)
            ensure(out.shape == want.shape, f"stride {s}, pad {p}: shape {out.shape}, {want.shape} bekleniyordu")
            close(out, want, what=f"stride {s}, pad {p}")
    R.test("conv2d_np: farklı stride/padding", cv)

    def ic():
        for s, p in [(1, 0), (2, 1)]:
            cols = need(ns, "im2col_np")(x, 3, 3, s, p)
            want = ref(x, w, np.zeros(4), s, p)
            ensure(cols.shape == (2, 27, want.shape[2] * want.shape[3]), f"im2col shape {cols.shape}")
            close((w.reshape(4, -1) @ cols).reshape(want.shape), want, what=f"W_mat @ im2col, stride {s}")
    R.test("im2col_np: W_mat @ cols = conv", ic)
