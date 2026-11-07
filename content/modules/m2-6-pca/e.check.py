import torch
from labcheck import close, no_loops, need, ensure, bench, fmt_time


def one(X, k):
    Xc = X - X.mean(0)
    U, s, Vh = torch.linalg.svd(Xc, full_matrices=False)
    V = Vh[:k]
    idx = V.abs().argmax(1, keepdim=True)
    return V * V.gather(1, idx).sign(), s[:k] ** 2 / (X.shape[0] - 1)


def check(ns, R):
    src = ns["__source__"]
    R.test("döngü yok", lambda: no_loops(src))
    g = torch.Generator().manual_seed(0)
    Xs = torch.randn(16, 40, 5, generator=g, dtype=torch.float64) * torch.tensor([4.0, 2, 1, 0.5, 0.1], dtype=torch.float64)

    def eq():
        P, var = need(ns, "batched_pca")(Xs, 3)
        ensure(tuple(P.shape) == (16, 3, 5) and tuple(var.shape) == (16, 3), f"shape'ler {tuple(P.shape)}, {tuple(var.shape)}")
        for b in range(16):
            Pb, vb = one(Xs[b], 3)
            close(P[b], Pb, atol=1e-9, what=f"veri seti {b} bileşenler"); close(var[b], vb, what=f"veri seti {b} varyanslar")
    R.test("16 veri seti: tek tek PCA ile aynı", eq)
    if "batched_pca" not in ns:
        return
    Xb = torch.randn(1000, 32, 8, generator=g)
    t_loop = bench(lambda: [one(x, 3) for x in Xb], repeat=3, number=1)
    t_vec = bench(lambda: ns["batched_pca"](Xb, 3), repeat=3)
    R.table("1000 küçük PCA (32 × 8)", ["yöntem", "süre", "döngüye göre"],
            [["for: 1000 ayrı SVD", fmt_time(t_loop), "1×"], ["senin batched_pca", fmt_time(t_vec), f"{t_loop / t_vec:,.1f}× hızlı"]])
