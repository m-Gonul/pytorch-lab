import torch
from labcheck import close, no_loops, need, ensure, bench, fmt_time


def check(ns, R):
    src = ns["__source__"]
    R.test("döngü yok", lambda: no_loops(src))
    g = torch.Generator().manual_seed(7)
    x = torch.randn(64, generator=g, dtype=torch.float64) * 3 + 1

    def bs():
        m = need(ns, "bootstrap_means")(x, 20000, torch.Generator().manual_seed(1))
        ensure(tuple(m.shape) == (20000,), f"shape {tuple(m.shape)}")
        ensure(float(m.min()) >= float(x.min()) and float(m.max()) <= float(x.max()), "ortalamalar verinin aralığı dışında")
        se = float(x.std(unbiased=False)) / 64 ** 0.5
        ensure(abs(float(m.mean()) - float(x.mean())) < 4 * se / 20000 ** 0.5 + 1e-3, "bootstrap ortalamalarının ortalaması x̄'dan uzak")
        ensure(abs(float(m.std()) / se - 1) < 0.05, f"bootstrap std {float(m.std()):.4f}, teorik σ̂/√n {se:.4f}")
        return f"bootstrap std {float(m.std()):.4f} ≈ σ̂/√n = {se:.4f}"
    R.test("bootstrap: standart hata ≈ σ̂/√n", bs)

    def det():
        a = need(ns, "bootstrap_means")(x, 50, torch.Generator().manual_seed(3))
        b = need(ns, "bootstrap_means")(x, 50, torch.Generator().manual_seed(3))
        ensure(torch.equal(a, b), "aynı seed farklı sonuç verdi: verilen generator kullanılmıyor")
    R.test("verilen generator kullanılıyor (tekrarlanabilir)", det)

    def bm():
        X = torch.randn(30, 12, generator=g, dtype=torch.float64)
        mu, var = need(ns, "batched_gaussian_mle")(X)
        close(mu, X.mean(1)); close(var, X.var(1, unbiased=False))
    R.test("batched_gaussian_mle: satır başına", bm)
    if "bootstrap_means" not in ns:
        return
    xb = torch.randn(500)

    def loop():
        n = xb.shape[0]
        gg = torch.Generator().manual_seed(0)
        return torch.stack([xb[torch.randint(n, (n,), generator=gg)].mean() for _ in range(2000)])
    t_loop = bench(loop, repeat=3, number=1)
    t_vec = bench(lambda: ns["bootstrap_means"](xb, 2000, torch.Generator().manual_seed(0)), repeat=3)
    R.table("2000 bootstrap örneği (n = 500)", ["yöntem", "süre", "döngüye göre"],
            [["for b: randint + mean", fmt_time(t_loop), "1×"],
             ["senin bootstrap_means", fmt_time(t_vec), f"{t_loop / t_vec:,.0f}× hızlı"]])
