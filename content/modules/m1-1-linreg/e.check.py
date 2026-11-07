import torch
from labcheck import close, max_loops, need, ensure, bench, fmt_time


def train_one(X, y, lr, steps):
    w = torch.zeros(X.shape[1], dtype=X.dtype, requires_grad=True)
    b = torch.zeros((), dtype=X.dtype, requires_grad=True)
    for _ in range(steps):
        loss = ((X @ w + b - y) ** 2).mean()
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad; b -= lr * b.grad
            w.grad = None; b.grad = None
    with torch.no_grad():
        return w.clone(), b.clone(), ((X @ w + b - y) ** 2).mean()


def check(ns, R):
    src = ns["__source__"]
    R.test("sweep içinde yalnız adım döngüsü", lambda: max_loops(src, 1, func="sweep", why="lr'ler üzerinde döngü olmamalı"))
    g = torch.Generator().manual_seed(0)
    X = torch.randn(150, 4, generator=g)
    y = X @ torch.tensor([1.0, -2.0, 0.5, 0.0]) + 1.0 + 0.1 * torch.randn(150, generator=g)
    lrs = torch.tensor([0.001, 0.01, 0.05, 0.1, 0.3])

    def eq():
        W, b, losses = need(ns, "sweep")(X, y, lrs, steps=60)
        ensure(tuple(W.shape) == (5, 4) and tuple(b.shape) == (5,) and tuple(losses.shape) == (5,), f"shape'ler {tuple(W.shape)}, {tuple(b.shape)}, {tuple(losses.shape)}")
        for i, lr in enumerate(lrs.tolist()):
            w1, b1, l1 = train_one(X, y, lr, 60)
            close(W[i], w1, rtol=1e-4, atol=1e-5, what=f"lr={lr} W")
            close(b[i], b1, rtol=1e-4, atol=1e-5, what=f"lr={lr} b")
            close(losses[i], l1, rtol=1e-4, atol=1e-5, what=f"lr={lr} kayıp")
        return "her lr tek tek eğitilmiş modelle birebir aynı"
    R.test("5 lr: tek tek eğitimle eşleşiyor", eq)
    if "sweep" not in ns:
        return
    lrs64 = torch.logspace(-3, -0.5, 64)
    t_loop = bench(lambda: [train_one(X, y, float(lr), 100) for lr in lrs64], repeat=3, number=1)
    t_vec = bench(lambda: ns["sweep"](X, y, lrs64, steps=100), repeat=3, number=1)
    _, _, Lf = ns["sweep"](X, y, lrs64, steps=100)
    best = int(torch.nan_to_num(Lf, nan=float("inf")).argmin())
    R.table("64 learning rate × 100 adım", ["yöntem", "süre", "döngüye göre"],
            [["for lr: 64 ayrı eğitim", fmt_time(t_loop), "1×"],
             ["senin sweep (tek tensör)", fmt_time(t_vec), f"{t_loop / t_vec:,.1f}× hızlı"]],
            note=f"En iyi lr ≈ {float(lrs64[best]):.3f} (son kayıp {float(Lf[best]):.4f}).")
