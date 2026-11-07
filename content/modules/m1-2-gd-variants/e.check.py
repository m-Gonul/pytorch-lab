import torch
from labcheck import close, max_loops, need, ensure, bench, fmt_time


def one(X, y, lr, beta, steps):
    w = torch.zeros(X.shape[1], dtype=X.dtype); v = torch.zeros_like(w)
    for _ in range(steps):
        g = 2 * X.T @ (X @ w - y) / X.shape[0]
        v = beta * v + g
        w = w - lr * v
    return ((X @ w - y) ** 2).mean()


def check(ns, R):
    src = ns["__source__"]
    R.test("sweep: yalnız adım döngüsü", lambda: max_loops(src, 1, func="sweep", why="kombinasyonlar üzerinde döngü olmamalı"))
    g = torch.Generator().manual_seed(0)
    X = (torch.randn(200, 2, generator=g) * torch.tensor([1.0, 8.0])).double()
    y = X @ torch.tensor([2.0, 0.3], dtype=torch.float64)
    lr_g, b_g = torch.meshgrid(torch.logspace(-4, -2.2, 6, dtype=torch.float64), torch.linspace(0, 0.95, 8, dtype=torch.float64), indexing="ij")
    lrs, betas = lr_g.reshape(-1), b_g.reshape(-1)

    def eq():
        L = need(ns, "sweep")(X, y, lrs, betas, 60)
        ensure(tuple(L.shape) == (48,), f"shape {tuple(L.shape)}")
        ref = torch.stack([one(X, y, float(a), float(b), 60) for a, b in zip(lrs, betas)])
        close(torch.nan_to_num(L, nan=1e30, posinf=1e30), torch.nan_to_num(ref, nan=1e30, posinf=1e30), rtol=1e-6, atol=1e-9)
        k = int(torch.nan_to_num(L, nan=float("inf")).argmin())
        return f"en iyi: lr = {float(lrs[k]):.2e}, β = {float(betas[k]):.2f}, kayıp {float(L[k]):.3g}"
    R.test("48 kombinasyon tek tek eğitimle aynı", eq)
    if "sweep" not in ns:
        return
    Xf, yf = X.float(), y.float()
    t_loop = bench(lambda: [one(Xf, yf, float(a), float(b), 100) for a, b in zip(lrs, betas)], repeat=3, number=1)
    t_vec = bench(lambda: ns["sweep"](Xf, yf, lrs.float(), betas.float(), 100), repeat=3, number=1)
    L = ns["sweep"](Xf, yf, lrs.float(), betas.float(), 100).view(6, 8)
    rows = [[f"{float(lr_g[i, 0]):.1e}"] + [("ıraksadı" if not torch.isfinite(L[i, j]) or L[i, j] > 1e6 else f"{float(L[i, j]):.2g}") for j in range(0, 8, 2)] for i in range(6)]
    R.table("100 adım sonra kayıp (satır: lr, sütun: β)", ["lr \\ β"] + [f"{float(b_g[0, j]):.2f}" for j in range(0, 8, 2)], rows,
            note="Momentum, kararlı kalan en büyük lr'yi düşürür ama o sınırın altında çok daha hızlı yakınsar.")
    R.table("48 (lr, β) çifti × 100 adım", ["yöntem", "süre", "döngüye göre"],
            [["for: 48 ayrı eğitim", fmt_time(t_loop), "1×"], ["senin sweep", fmt_time(t_vec), f"{t_loop / t_vec:,.0f}× hızlı"]])
