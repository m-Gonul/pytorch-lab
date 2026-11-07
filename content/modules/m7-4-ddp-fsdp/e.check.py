import torch
from labcheck import close, need, ensure


def reference(ns, steps, lr=1e-2):
    m = ns["make_model"]()
    X, y = ns["make_data"]()
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for _ in range(steps):
        loss = ((m(X) - y) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return [p.detach() for p in m.parameters()], sum(p.numel() for p in m.parameters())


def check(ns, R):
    def fl():
        ts = [torch.randn(3, 4), torch.randn(5), torch.randn(2, 1, 2)]
        f = need(ns, "flatten")(ts)
        ensure(f.shape == (21,), "düz vektör 21 eleman")
        for a, b in zip(need(ns, "unflatten")(f, ts), ts):
            close(a, b)
    R.test("flatten / unflatten", fl)

    def z(world):
        def f():
            P, sizes = need(ns, "run_zero1")(world, 10)
            ref, n = reference(ns, 10)
            for r in range(world):
                for a, b in zip(P[r], ref):
                    close(a, b, atol=1e-5, what=f"işçi {r}")
            per = -(-n // world) * 2
            ensure(all(s == per for s in sizes), f"işçi başına durum {sizes}; {per} bekleniyordu (2 × ⌈{n}/{world}⌉)")
            R.table(f"ZeRO-1, {world} işçi", ["", "eleman"], [["parametre", str(n)], ["Adam durumu, tek işçi", str(2 * n)], ["Adam durumu, işçi başına", str(per)]])
        return f
    R.test("ZeRO-1 (2 işçi) = torch.optim.Adam", z(2))
    R.test("ZeRO-1 (4 işçi; 1377 parametre 4'e bölünmüyor → dolgu)", z(4))
