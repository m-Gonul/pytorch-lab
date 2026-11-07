import torch
from labcheck import close, max_loops, need, ensure, bench, fmt_time


def circles(g, n):
    r = torch.cat([torch.full((n,), 1.0), torch.full((n,), 3.0)]) + 0.2 * torch.randn(2 * n, generator=g)
    t = torch.rand(2 * n, generator=g) * 6.283185
    return torch.stack([r * t.cos(), r * t.sin()], 1), torch.cat([torch.ones(n), -torch.ones(n)])


def check(ns, R):
    src = ns["__source__"]
    R.test("rbf_kernel/predict_kernel: döngü yok", lambda: (max_loops(src, 0, func="rbf_kernel"), max_loops(src, 0, func="predict_kernel")))
    R.test("fit_kernel_svm: yalnız adım döngüsü", lambda: max_loops(src, 1, func="fit_kernel_svm"))
    g = torch.Generator().manual_seed(0)
    A, B = torch.randn(13, 3, generator=g, dtype=torch.float64), torch.randn(7, 3, generator=g, dtype=torch.float64)
    R.test("rbf_kernel doğru", lambda: close(need(ns, "rbf_kernel")(A, B, 0.7), torch.exp(-0.7 * torch.cdist(A, B) ** 2), rtol=1e-9, atol=1e-12))

    def circ():
        X, y = circles(g, 120)
        Xt, yt = circles(g, 100)
        alpha, b = need(ns, "fit_kernel_svm")(X, y, gamma=0.5)
        acc = (need(ns, "predict_kernel")(X, alpha, b, 0.5, Xt) == yt).float().mean().item()
        ensure(acc >= 0.95, f"test doğruluğu {acc:.2f}")
        w = torch.linalg.lstsq(torch.cat([X, torch.ones(240, 1)], 1), y[:, None]).solution
        lin = ((torch.cat([Xt, torch.ones(200, 1)], 1) @ w).sign().view(-1) == yt).float().mean().item()
        nsv = int((alpha.abs() > 1e-6).sum())
        return f"iç içe çemberler: kernel SVM test doğruluğu {acc:.3f}, lineer model {lin:.3f}"
    R.test("kernel SVM doğrusal olmayan sınırı öğreniyor", circ)
    if "rbf_kernel" not in ns:
        return
    Xb = torch.randn(1500, 10, generator=g)
    t_loop = bench(lambda: torch.stack([torch.exp(-0.5 * ((Xb - x) ** 2).sum(1)) for x in Xb]), repeat=3, number=1)
    t_vec = bench(lambda: ns["rbf_kernel"](Xb, Xb, 0.5), repeat=3)
    R.table("1500 × 1500 RBF Gram matrisi", ["yöntem", "süre", "döngüye göre"],
            [["for satır: exp(−γ‖x − X‖²)", fmt_time(t_loop), "1×"], ["senin rbf_kernel", fmt_time(t_vec), f"{t_loop / t_vec:,.0f}× hızlı"]],
            note="Kernel yöntemlerinin darboğazı n × n matris: n = 10⁵ için 40 GB (float32). Büyük veride yaklaşık yöntemler (Nyström, random features) kullanılır.")
