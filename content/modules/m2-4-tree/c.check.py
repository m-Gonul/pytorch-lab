import numpy as np
from labcheck import close, no_loops, no_imports, need, ensure


def brute(X, y, C):
    n, d = X.shape
    best = None
    for j in range(d):
        v = np.unique(X[:, j])
        for t in (v[:-1] + v[1:]) / 2:
            m = X[:, j] <= t
            def G(s):
                p = np.bincount(s, minlength=C) / len(s)
                return 1 - (p ** 2).sum()
            s = m.mean() * G(y[m]) + (1 - m.mean()) * G(y[~m])
            if best is None or s < best[2] - 1e-12:
                best = (j, t, s)
    return best


def check(ns, R):
    src = ns["__source__"]
    R.test("döngü yok", lambda: no_loops(src))
    R.test("torch/sklearn kullanılmıyor", lambda: no_imports(src, {"torch", "sklearn"}))
    f = lambda *a: need(ns, "best_split_np")(*a)
    R.test("elle örnek (0, 2.5, 0.25)", lambda: close(f(np.array([[1.0], [2], [3], [4], [5], [6]]), np.array([0, 0, 1, 1, 1, 0]), 2), [0, 2.5, 0.25]))
    R.test("bölünemezse None", lambda: ensure(f(np.array([[1.0], [1.0]]), np.array([0, 1]), 2) is None, "None bekleniyordu"))
    rng = np.random.default_rng(0)

    def rnd():
        for C, d in [(2, 3), (4, 5)]:
            X = rng.normal(size=(60, d)).round(2)
            y = rng.integers(0, C, 60)
            got, want = f(X, y, C), brute(X, y, C)
            ensure(got[0] == want[0], f"C={C}: özellik {got[0]}, {want[0]} bekleniyordu")
            close(got[1:], want[1:], what=f"C={C}")
    R.test("rastgele veri (tekrarlı değerler dahil): kaba kuvvetle aynı", rnd)
