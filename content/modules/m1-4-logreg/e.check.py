import torch
from labcheck import close, max_loops, need, ensure, bench, fmt_time


def binary(X, t, lr, steps):
    w = torch.zeros(X.shape[1]); b = torch.zeros(())
    for _ in range(steps):
        r = torch.sigmoid(X @ w + b) - t
        w = w - lr * X.T @ r / len(t)
        b = b - lr * r.mean()
    return w, b


def check(ns, R):
    src = ns["__source__"]
    R.test("train_ovr içinde yalnız adım döngüsü", lambda: max_loops(src, 1, func="train_ovr", why="sınıflar üzerinde döngü olmamalı"))
    R.test("predict_ovr içinde döngü yok", lambda: max_loops(src, 0, func="predict_ovr"))
    g = torch.Generator().manual_seed(3)
    centers = torch.tensor([[0.0, 3.0], [-3.0, -2.0], [3.0, -2.0], [0.0, -4.5]])
    y = torch.arange(4).repeat_interleave(60)
    X = centers[y] + torch.randn(240, 2, generator=g)

    def eq():
        W, b = need(ns, "train_ovr")(X, y, 4, lr=0.5, steps=80)
        ensure(tuple(W.shape) == (2, 4) and tuple(b.shape) == (4,), f"shape'ler {tuple(W.shape)}, {tuple(b.shape)}")
        for k in range(4):
            wk, bk = binary(X, (y == k).float(), 0.5, 80)
            close(W[:, k], wk, rtol=1e-4, atol=1e-5, what=f"sınıf {k} ağırlıkları")
            close(b[k], bk, rtol=1e-4, atol=1e-5, what=f"sınıf {k} bias")
        return "4 sütunun her biri ayrı eğitilmiş ikili modelle aynı"
    R.test("K ayrı ikili modelle eşleşiyor", eq)

    def acc():
        W, b = need(ns, "train_ovr")(X, y, 4)
        a = (need(ns, "predict_ovr")(X, W, b) == y).float().mean().item()
        ensure(a >= 0.85, f"doğruluk {a:.2f}")
        return f"4 sınıf, doğruluk {a:.3f}"
    R.test("predict_ovr doğruluğu ≥ 0.85", acc)
    if "train_ovr" not in ns:
        return
    K = 10
    Xb = torch.randn(2000, 20, generator=g)
    yb = torch.randint(0, K, (2000,), generator=g)
    t_loop = bench(lambda: [binary(Xb, (yb == k).float(), 0.5, 100) for k in range(K)], repeat=3, number=1)
    t_vec = bench(lambda: ns["train_ovr"](Xb, yb, K, lr=0.5, steps=100), repeat=3, number=1)
    R.table(f"{K} sınıf × 100 adım (n = 2000, d = 20)", ["yöntem", "süre", "döngüye göre"],
            [["for k: K ayrı ikili model", fmt_time(t_loop), "1×"],
             ["senin train_ovr", fmt_time(t_vec), f"{t_loop / t_vec:,.1f}× hızlı"]],
            note="Kazancın kaynağı: K adet (n×d)·(d,) çarpımı yerine tek (n×d)·(d×K) matris çarpımı.")
