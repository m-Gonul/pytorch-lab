import torch
from labcheck import close, need, ensure


def check(ns, R_):
    dev = ns.get("DEVICE", "cpu")
    g = torch.Generator().manual_seed(0)

    def sc():
        for n in (1, 1000, 1024, 1025, 70000):
            x = torch.randint(-3, 4, (n,), generator=g).float().to(dev)
            close(need(ns, "scan")(x).cpu().numpy(), torch.cumsum(x.cpu().double(), 0).numpy(), atol=1e-3, what=f"n = {n}")
        x = torch.ones(5000).to(dev)
        close(ns["scan"](x, 64).cpu().numpy(), torch.arange(1, 5001).double().numpy(), what="küçük blok: 3 seviyeli özyineleme")
    R_.test("hiyerarşik tarama", sc)

    def lr():
        a = (torch.rand(4, 300, generator=g) * 0.5 + 0.5).to(dev)
        b = torch.randn(4, 300, generator=g).to(dev)
        x = need(ns, "linear_recurrence")(a, b).cpu().double()
        ref = torch.zeros(4, 300, dtype=torch.float64)
        acc = torch.zeros(4, dtype=torch.float64)
        for t in range(300):
            acc = a[:, t].cpu().double() * acc + b[:, t].cpu().double()
            ref[:, t] = acc
        close(x.numpy(), ref.numpy(), atol=1e-4, what="tl.associative_scan ile afin tarama")
        R_.note("tl.associative_scan herhangi bir birleşmeli işlecin (burada demetler üzerinde) blok içi taramasını derler; Mamba'nın seçici taraması aynı yapıdadır (köşegen a_t, giriş-bağımlı).")
    R_.test("doğrusal özyineleme", lr)
