import torch
from labcheck import close, no_loops, need, ensure, bench, fmt_time


def check(ns, R):
    src = ns["__source__"]
    R.test("döngü yok", lambda: no_loops(src))
    g = torch.Generator().manual_seed(0)
    P = torch.randn(40, 2, generator=g, dtype=torch.float64)
    x, y = P[:, 0], P[:, 1]

    def bg():
        G = need(ns, "batch_grad")(P)
        ensure(tuple(G.shape) == (40, 2), f"shape {tuple(G.shape)}, (40, 2) bekleniyordu")
        close(G, torch.stack([y, x + 2 * y], 1), what="∇f = (y, x + 2y)")
    R.test("batch_grad: 40 noktada (y, x + 2y)", bg)

    def bj():
        J = need(ns, "batch_jacobian")(P)
        ensure(tuple(J.shape) == (40, 2, 2), f"shape {tuple(J.shape)}, (40, 2, 2) bekleniyordu")
        want = torch.stack([torch.stack([y, x], 1), torch.stack([torch.ones_like(x), 2 * y], 1)], 1)
        close(J, want, what="J = [[y, x], [1, 2y]]")
    R.test("batch_jacobian: (40, 2, 2)", bj)
    if "batch_jacobian" not in ns:
        return
    from torch.autograd.functional import jacobian
    gv = lambda p: torch.stack([p[0] * p[1], p[0] + p[1] ** 2])
    Pb = torch.randn(512, 2)
    t_loop = bench(lambda: torch.stack([jacobian(gv, p) for p in Pb]), repeat=3, number=1)
    t_vm = bench(lambda: ns["batch_jacobian"](Pb), repeat=3)
    R.table("512 noktanın Jacobian'ı", ["yöntem", "süre", "döngüye göre"],
            [["for + autograd.functional.jacobian", fmt_time(t_loop), "1×"],
             ["senin batch_jacobian", fmt_time(t_vm), f"{t_loop / t_vm:,.0f}× hızlı"]],
            note="Döngüde her nokta için ayrı bir grafik kurulur ve m = 2 ayrı backward yapılır; vmap tüm bunları batch boyutlu tek işlemlere çevirir.")
