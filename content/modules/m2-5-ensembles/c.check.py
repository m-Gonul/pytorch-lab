import numpy as np
from labcheck import close, max_loops, no_imports, need, ensure


def brute(X, r):
    n, d = X.shape
    best = None
    for j in range(d):
        v = np.unique(X[:, j])
        for t in (v[:-1] + v[1:]) / 2:
            m = X[:, j] <= t
            g = r[m].sum() ** 2 / m.sum() + r[~m].sum() ** 2 / (~m).sum()
            if best is None or g > best[0] + 1e-12:
                best = (g, j, t, r[m].mean(), r[~m].mean())
    return best[1:]


def check(ns, R):
    src = ns["__source__"]
    R.test("best_reg_stump: döngü yok", lambda: max_loops(src, 0, func="best_reg_stump"))
    R.test("gbm_np / gbm_predict_np: yalnız tur döngüsü", lambda: (max_loops(src, 1, func="gbm_np"), max_loops(src, 1, func="gbm_predict_np")))
    R.test("torch/sklearn kullanılmıyor", lambda: no_imports(src, {"torch", "sklearn"}))
    rng = np.random.default_rng(0)

    def st():
        X = rng.normal(size=(50, 4)).round(2); r = rng.normal(size=50)
        got, want = need(ns, "best_reg_stump")(X, r), brute(X, r)
        ensure(got[0] == want[0], f"özellik {got[0]}, {want[0]} bekleniyordu"); close(got[1:], want[1:])
        close(need(ns, "best_reg_stump")(np.array([[1.], [2], [3], [4]]), np.array([-1., -1, 1, 1]))[1:], [2.5, -1, 1], what="elle örnek")
    R.test("best_reg_stump kaba kuvvetle aynı", st)

    def fit():
        X = rng.uniform(-2, 2, size=(600, 3)); y = np.sin(2 * X[:, 0]) + X[:, 1] ** 2 + 0.05 * rng.normal(size=600)
        Xt = rng.uniform(-2, 2, size=(300, 3)); yt = np.sin(2 * Xt[:, 0]) + Xt[:, 1] ** 2
        m = need(ns, "gbm_np")(X, y, 200, 0.2)
        p = need(ns, "gbm_predict_np")(m, Xt)
        r2 = 1 - np.mean((p - yt) ** 2) / np.var(yt)
        ensure(r2 >= 0.9, f"test R² = {r2:.3f}")
        return f"test R² = {r2:.3f} (stump'ların toplamı: toplamsal model)"
    R.test("doğrusal olmayan fonksiyonda test R² ≥ 0.9", fit)
