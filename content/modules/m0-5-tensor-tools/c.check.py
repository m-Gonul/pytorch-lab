import numpy as np
from labcheck import close, no_loops, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("döngü yok", lambda: no_loops(src))
    R.test("torch kullanılmıyor", lambda: no_imports(src, {"torch"}))
    x = np.random.default_rng(0).normal(size=50)

    def sw():
        W = need(ns, "sliding_windows")(x, 7)
        ensure(W.shape == (44, 7), f"shape {W.shape}")
        ensure(np.shares_memory(W, x), "pencereler kopya; x ile bellek paylaşılmalı")
        close(W, np.array([x[i:i + 7] for i in range(44)]))
        return f"strides (bayt) = {W.strides}"
    R.test("sliding_windows: doğru ve kopyasız", sw)
    R.test("moving_average", lambda: close(need(ns, "moving_average")(x, 5), np.convolve(x, np.ones(5) / 5, mode="valid")))

    def iv():
        f = need(ns, "is_view_of")
        A = np.arange(12.0).reshape(3, 4)
        ensure(f(A.T, A) and f(A[:, 1], A) and not f(A.copy(), A) and not f(A[[0, 2]], A), "view/kopya ayrımı yanlış (fancy indexing kopya yapar)")
        return "A.T ve A[:, 1] view; A.copy() ve A[[0, 2]] kopya"
    R.test("is_view_of: view vs kopya", iv)
