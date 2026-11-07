import numpy as np
from labcheck import close, no_loops, no_imports, need


def check(ns, R):
    src = ns["__source__"]
    R.test("döngü/comprehension yok", lambda: no_loops(src))
    R.test("torch kullanılmıyor", lambda: no_imports(src, {"torch"}))
    rng = np.random.default_rng(0)

    def mm():
        f = need(ns, "matmul_np")
        close(f(np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])), [[19, 22], [43, 50]])
        A, B = rng.normal(size=(7, 5)), rng.normal(size=(5, 3))
        out = f(A, B)
        assert isinstance(out, np.ndarray), "np.ndarray döndürmeli"
        close(out, A @ B, what="(7×5)(5×3)")
    R.test("matmul_np doğru", mm)

    def cc():
        f = need(ns, "center_columns")
        X = rng.normal(loc=3.0, size=(6, 4))
        out = f(X.copy())
        close(out, X - X.mean(0), what="center_columns")
        close(np.asarray(out).mean(0), np.zeros(4), atol=1e-9, what="sütun ortalamaları")
    R.test("center_columns: sütun ortalamaları 0", cc)
