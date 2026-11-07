import torch
from labcheck import close, need, ensure, uses_any


def mean_update(X, C):
    lab = ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)
    C = C.clone()
    for k in range(len(C)):
        if (lab == k).any():
            C[k] = X[lab == k].mean(0)
    return C


def check(ns, R):
    src = ns["__source__"]
    g = torch.Generator().manual_seed(0)
    X = torch.cat([torch.randn(30, 3, generator=g, dtype=torch.float64) + c for c in (0.0, 4.0, -4.0)])

    def km():
        C0 = X[[0, 1, 2]].clone()
        C, lab, J = need(ns, "kmeans_torch")(X, C0, 10)
        Cr = C0.clone()
        for _ in range(10):
            Cr = mean_update(X, Cr)
        close(C, Cr)
        close(J, ((X - Cr[lab]) ** 2).sum(), what="inertia")
    R.test("kmeans_torch referans Lloyd ile aynı", km)
    R.test("autograd kullanılıyor", lambda: ensure(uses_any(src, ["grad", "backward"]), "lloyd_step_via_grad autograd kullanmalı"))

    def ng():
        f = need(ns, "lloyd_step_via_grad")
        close(f(torch.tensor([[0.0, 0], [1, 0], [4, 4], [5, 4]], dtype=torch.float64), torch.tensor([[0.0, 0], [1, 0]], dtype=torch.float64)), [[0, 0], [10 / 3, 8 / 3]], what="elle örnek")
        C = torch.randn(4, 3, generator=g, dtype=torch.float64) * 3
        C[3] = 100.0
        close(f(X, C), mean_update(X, C), what="rastgele merkezler + boş küme")
        return "Newton adımı c − ∇J/(2nₖ) = küme ortalaması"
    R.test("lloyd_step_via_grad = ortalama güncellemesi", ng)
