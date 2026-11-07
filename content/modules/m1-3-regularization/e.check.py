import torch
from labcheck import close, max_loops, need, ensure, bench, fmt_time


def solve_one(X, y, lam):
    xm, ym = X.mean(0), y.mean()
    Xc = X - xm
    w = torch.linalg.solve(Xc.T @ Xc + lam * torch.eye(X.shape[1], dtype=X.dtype), Xc.T @ (y - ym))
    return w, ym - xm @ w


def check(ns, R):
    src = ns["__source__"]
    R.test("ridge_path_torch: döngü yok", lambda: max_loops(src, 0, func="ridge_path_torch"))
    R.test("cv_ridge: yalnız fold döngüsü", lambda: max_loops(src, 1, func="cv_ridge", why="λ'lar üzerinde döngü olmamalı"))
    g = torch.Generator().manual_seed(0)
    X = torch.randn(150, 20, generator=g, dtype=torch.float64)
    y = X[:, :4] @ torch.tensor([1.0, -1.0, 2.0, 0.5], dtype=torch.float64) + 3 + torch.randn(150, generator=g, dtype=torch.float64)
    lams = torch.logspace(-3, 3, 30, dtype=torch.float64)

    def path():
        W, b = need(ns, "ridge_path_torch")(X, y, lams)
        ensure(tuple(W.shape) == (30, 20) and tuple(b.shape) == (30,), f"shape'ler {tuple(W.shape)}, {tuple(b.shape)}")
        for i in (0, 10, 29):
            w1, b1 = solve_one(X, y, float(lams[i]))
            close(W[i], w1, rtol=1e-7, atol=1e-9, what=f"λ = {float(lams[i]):.3g} w"); close(b[i], b1, rtol=1e-7, what="b")
    R.test("ridge_path_torch: her λ'da kapalı form", path)

    def cv():
        mse, best = need(ns, "cv_ridge")(X, y, lams, k=5)
        ref = torch.zeros(30, dtype=torch.float64)
        idx = torch.arange(150)
        for i in range(5):
            va, tr = idx[i::5], idx[idx % 5 != i]
            for j, lam in enumerate(lams):
                w, b = solve_one(X[tr], y[tr], float(lam))
                ref[j] += ((X[va] @ w + b - y[va]) ** 2).mean() / 5
        close(mse, ref, rtol=1e-7)
        close(best, lams[ref.argmin()])
        return f"en iyi λ = {float(best):.3g}, CV MSE {float(mse.min()):.3f} (λ → 0: {float(mse[0]):.3f}, λ → ∞: {float(mse[-1]):.3f})"
    R.test("cv_ridge: referans CV ile aynı", cv)
    if "ridge_path_torch" not in ns:
        return
    Xb = torch.randn(1000, 100, generator=g, dtype=torch.float64)
    yb = Xb @ torch.randn(100, generator=g, dtype=torch.float64)
    lb = torch.logspace(-3, 3, 200, dtype=torch.float64)
    t_loop = bench(lambda: [solve_one(Xb, yb, float(l)) for l in lb], repeat=3, number=1)
    t_svd = bench(lambda: ns["ridge_path_torch"](Xb, yb, lb), repeat=3)
    R.table("200 λ, n = 1000, d = 100", ["yöntem", "süre", "döngüye göre"],
            [["for λ: solve (200 × O(d³))", fmt_time(t_loop), "1×"],
             ["senin ridge_path_torch (1 SVD)", fmt_time(t_svd), f"{t_loop / t_svd:,.0f}× hızlı"]])
