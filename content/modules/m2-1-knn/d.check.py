import torch
from labcheck import close, need, ensure, max_loops


def check(ns, R):
    src = ns["__source__"]
    R.test("döngü yok", lambda: (max_loops(src, 0, func="sqdist_torch"), max_loops(src, 0, func="knn_predict_torch")))
    g = torch.Generator().manual_seed(2)
    A, B = torch.randn(30, 8, generator=g, dtype=torch.float64), torch.randn(50, 8, generator=g, dtype=torch.float64)
    R.test("sqdist_torch = cdist²", lambda: close(need(ns, "sqdist_torch")(A, B), torch.cdist(A, B) ** 2, rtol=1e-9, atol=1e-9))
    R.test("sqdist_torch(A, A) ≥ 0", lambda: ensure(bool((need(ns, "sqdist_torch")(A.float(), A.float()) >= 0).all()), "negatif değer var; clamp_min(0) eksik"))

    def acc():
        centers = torch.tensor([[0.0, 3.0], [-3.0, -2.0], [3.0, -2.0]])
        ytr = torch.arange(3).repeat_interleave(80)
        Xtr = centers[ytr] + torch.randn(240, 2, generator=g)
        yte = torch.arange(3).repeat_interleave(30)
        Xte = centers[yte] + torch.randn(90, 2, generator=g)
        p = need(ns, "knn_predict_torch")(Xte, Xtr, ytr, 7, 3)
        ensure(tuple(p.shape) == (90,), f"shape {tuple(p.shape)}")
        a = (p == yte).float().mean().item()
        ensure(a >= 0.85, f"test doğruluğu {a:.2f}")
        D = torch.cdist(Xte, Xtr) ** 2
        idx = D.argsort(1)[:, :7]
        ref = torch.nn.functional.one_hot(ytr[idx], 3).sum(1).argmax(1)
        ensure(bool((p == ref).all()), "referans k-NN ile farklı tahminler")
        return f"3 sınıf, k=7, test doğruluğu {a:.3f}"
    R.test("knn_predict_torch: referansla aynı, doğruluk ≥ 0.85", acc)
