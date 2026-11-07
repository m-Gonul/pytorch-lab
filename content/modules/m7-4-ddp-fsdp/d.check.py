import torch
from labcheck import close, need, ensure


def reference(ns, steps):
    torch.manual_seed(0)
    m = ns["make_model"]()
    X, y = ns["make_data"]()
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    for _ in range(steps):
        loss = ((m(X) - y) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return [p.detach() for p in m.parameters()]


def check(ns, R):
    need(ns, "make_model"); need(ns, "make_data")

    def ddp():
        P = need(ns, "run")(2, 5, False)
        ensure(len(P) == 2, "iki işçinin parametreleri")
        for a, b in zip(P[0], P[1]):
            close(a, b, atol=0, rtol=0, what="iki işçi birebir aynı")
        for a, b in zip(P[0], reference(ns, 5)):
            close(a, b, atol=1e-6, what="tek süreçli tam batch ile")
    R.test("DDP (2 süreç) = tek süreçli tam batch", ddp)

    def manual():
        P = need(ns, "run")(2, 5, True)
        for a, b in zip(P[0], reference(ns, 5)):
            close(a, b, atol=1e-6)
    R.test("DDP'siz, elle all-reduce", manual)

    def four():
        P = need(ns, "run")(4, 3, True)
        for r in range(1, 4):
            for a, b in zip(P[0], P[r]):
                close(a, b, atol=0, rtol=0, what=f"işçi {r}")
        for a, b in zip(P[0], reference(ns, 3)):
            close(a, b, atol=1e-6)
        return "4 süreç, 16'şar örnek: tek süreçli 64'lük batch ile aynı"
    R.test("4 süreç", four)
