import random
from labcheck import close, no_imports, need, ensure


def ref_lasso(X, y, lam, iters):
    n, d = len(X), len(X[0])
    w = [0.0] * d
    for _ in range(iters):
        for j in range(d):
            rho = sum(X[i][j] * (y[i] - sum(X[i][k] * w[k] for k in range(d) if k != j)) for i in range(n)) / n
            z = sum(X[i][j] ** 2 for i in range(n)) / n
            s = rho - lam if rho > lam else (rho + lam if rho < -lam else 0.0)
            w[j] = s / z
    return w


def check(ns, R):
    src = ns["__source__"]
    R.test("numpy/torch kullanılmıyor", lambda: no_imports(src, {"numpy", "torch"}))
    R.test("soft_threshold", lambda: close([need(ns, "soft_threshold")(z, 0.5) for z in (1.5, -0.3, -2, 0.5, -0.5)], [1, 0, -1.5, 0, 0]))
    X, y = [[1, 0], [0, 1], [1, 1]], [1, 2, 3]
    R.test("ridge2: OLS (1, 2) ve λ = 1 (0.875, 1.375)", lambda: (close(need(ns, "ridge2")(X, y, 0.0), [1, 2]), close(need(ns, "ridge2")(X, y, 1.0), [0.875, 1.375])))
    rng = random.Random(0)
    Xr = [[rng.gauss(0, 1) for _ in range(4)] for _ in range(30)]
    wt = [2.0, 0.0, -1.0, 0.0]
    yr = [sum(a * b for a, b in zip(x, wt)) + rng.gauss(0, 0.1) for x in Xr]

    def lc():
        w = need(ns, "lasso_cd")(Xr, yr, 0.1, 50)
        close(w, ref_lasso(Xr, yr, 0.1, 50), rtol=1e-6, atol=1e-9)
        return "katsayılar: " + ", ".join(f"{v:.3f}" for v in w)
    R.test("lasso_cd referans coordinate descent ile aynı", lc)

    def sparse():
        w = need(ns, "lasso_cd")(Xr, yr, 0.3, 50)
        zeros = sum(v == 0.0 for v in w)
        ensure(zeros >= 2, f"λ = 0.3'te yalnız {zeros} katsayı tam sıfır; gerçek modelde 2 sıfır var")
        w_big = need(ns, "lasso_cd")(Xr, yr, 100.0, 5)
        ensure(all(v == 0.0 for v in w_big), "çok büyük λ'da tüm katsayılar tam 0 olmalı")
        return f"λ = 0.3 → {zeros} tam sıfır; gerçek ağırlıklar [2, 0, −1, 0]"
    R.test("lasso tam sıfırlar üretiyor", sparse)
