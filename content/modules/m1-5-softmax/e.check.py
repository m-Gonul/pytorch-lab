import torch
from torch.func import jacrev, vmap
from labcheck import close, no_loops, need, ensure, bench, fmt_time


def check(ns, R):
    src = ns["__source__"]
    R.test("döngü yok", lambda: no_loops(src))
    g = torch.Generator().manual_seed(0)
    Z = torch.randn(32, 6, generator=g, dtype=torch.float64)

    def jac():
        J = need(ns, "softmax_jacobian")(Z)
        ensure(tuple(J.shape) == (32, 6, 6), f"shape {tuple(J.shape)}")
        close(J, vmap(jacrev(lambda z: torch.softmax(z, 0)))(Z), what="vmap(jacrev(softmax)) ile")
        close(J.sum(2), torch.zeros(32, 6, dtype=torch.float64), atol=1e-12, what="satır toplamları (Σ p = 1 olduğundan 0)")
        return "J simetrik ve satır toplamları 0"
    R.test("softmax_jacobian = vmap(jacrev(softmax))", jac)

    def cm():
        f = need(ns, "confusion_matrix")
        C = f(torch.tensor([0, 0, 1, 2, 2]), torch.tensor([0, 1, 1, 2, 0]), 3)
        ensure(C.tolist() == [[1, 1, 0], [0, 1, 0], [1, 0, 1]], f"{C.tolist()}")
        yt = torch.randint(0, 7, (500,), generator=g); yp = torch.randint(0, 7, (500,), generator=g)
        ref = torch.zeros(7, 7, dtype=torch.long)
        for a, b in zip(yt.tolist(), yp.tolist()):
            ref[a, b] += 1
        ensure(torch.equal(f(yt, yp, 7).long(), ref), "rastgele 7 sınıfta yanlış")
        C2 = f(torch.tensor([0, 0]), torch.tensor([0, 0]), 4)
        ensure(tuple(C2.shape) == (4, 4), "hiç görünmeyen sınıflar da matriste olmalı (minlength)")
    R.test("confusion_matrix: elle örnek, rastgele, eksik sınıflar", cm)
    if "softmax_jacobian" not in ns or "confusion_matrix" not in ns:
        return
    Zb = torch.randn(2000, 10, generator=g)
    yt = torch.randint(0, 10, (100_000,), generator=g); yp = torch.randint(0, 10, (100_000,), generator=g)

    def loopJ():
        out = []
        for z in Zb:
            p = torch.softmax(z, 0)
            out.append(torch.diag(p) - torch.outer(p, p))
        return torch.stack(out)

    def loopC():
        C = [[0] * 10 for _ in range(10)]
        for a, b in zip(yt.tolist(), yp.tolist()):
            C[a][b] += 1
        return C
    tJl, tJv = bench(loopJ, repeat=3, number=1), bench(lambda: ns["softmax_jacobian"](Zb), repeat=3)
    tCl, tCv = bench(loopC, repeat=3, number=1), bench(lambda: ns["confusion_matrix"](yt, yp, 10), repeat=3)
    R.table("Döngü vs vektörize", ["iş", "döngü", "vektörize", "kazanç"],
            [["2000 softmax Jacobian'ı (K = 10)", fmt_time(tJl), fmt_time(tJv), f"{tJl / tJv:,.0f}×"],
             ["confusion matrix, 100 000 örnek", fmt_time(tCl), fmt_time(tCv), f"{tCl / tCv:,.0f}×"]])
