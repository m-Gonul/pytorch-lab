import copy
import torch
import torch.nn.functional as F
from labcheck import close, need, ensure


def check(ns, R):
    torch.manual_seed(0)
    base = torch.nn.Sequential(torch.nn.Linear(20, 64), torch.nn.ReLU(), torch.nn.Linear(64, 5)).double()
    x, y = torch.randn(70, 20, dtype=torch.float64), torch.randint(0, 5, (70,))

    def eq(opt_cls, **kw):
        def f():
            a, b = copy.deepcopy(base), copy.deepcopy(base)
            oa, ob = opt_cls(a.parameters(), **kw), opt_cls(b.parameters(), **kw)
            for _ in range(3):
                need(ns, "accum_step")(a, oa, x, y, micro=16)
                F.cross_entropy(b(x), y).backward(); ob.step(); ob.zero_grad()
            for p, q in zip(a.parameters(), b.parameters()):
                close(p, q, atol=1e-10)
            ensure(all(p.grad is None or p.grad.abs().sum() == 0 for p in a.parameters()), "adımdan sonra gradyanlar sıfırlanmalı")
        return f
    R.test("SGD: 5 mikro-batch (16+16+16+16+6) = tek batch", eq(torch.optim.SGD, lr=0.1))
    R.test("Adam: 3 adım, aynı parametreler", eq(torch.optim.Adam, lr=1e-2))

    def bn():
        torch.manual_seed(0)
        m = torch.nn.Sequential(torch.nn.Linear(20, 32), torch.nn.BatchNorm1d(32), torch.nn.ReLU(), torch.nn.Linear(32, 5)).double()
        a, b = copy.deepcopy(m), copy.deepcopy(m)
        oa, ob = torch.optim.SGD(a.parameters(), lr=0.1), torch.optim.SGD(b.parameters(), lr=0.1)
        need(ns, "accum_step")(a, oa, x, y, micro=16)
        F.cross_entropy(b(x), y).backward(); ob.step()
        diff = max((p - q).abs().max().item() for p, q in zip(a.parameters(), b.parameters()))
        ensure(diff > 1e-6, "BatchNorm'lu modelde fark bekleniyordu")
        return f"BatchNorm'lu modelde en büyük parametre farkı {diff:.2e}: istatistikler mikro-batch üzerinden"
    R.test("BatchNorm istisnası", bn)

    def mem():
        m = torch.nn.Sequential(torch.nn.Linear(256, 1024), torch.nn.GELU(), torch.nn.Linear(1024, 10))
        opt = torch.optim.SGD(m.parameters(), lr=0.0)
        xb, yb = torch.randn(512, 256), torch.randint(0, 10, (512,))
        rows = []
        for micro in (512, 128, 32):
            peak = [0]
            cur = [0]

            def pack(t):
                cur[0] += t.numel() * t.element_size(); peak[0] = max(peak[0], cur[0]); return t

            def unpack(t):
                return t
            orig_backward = torch.Tensor.backward

            def bw(self, *a, **k):
                cur[0] = 0
                return orig_backward(self, *a, **k)
            torch.Tensor.backward = bw
            try:
                with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
                    need(ns, "accum_step")(m, opt, xb, yb, micro)
            finally:
                torch.Tensor.backward = orig_backward
            rows.append([str(micro), f"{512 // micro}", f"{peak[0]:,}"])
        ensure(int(rows[-1][2].replace(",", "")) < int(rows[0][2].replace(",", "")), "küçük mikro-batch daha az tepe bellek")
        R.table("Mikro-batch başına tepe saklanan bayt (B = 512)", ["mikro-batch", "geçiş", "tepe bayt"], rows,
                note="Ağırlıklar her mikro-batch'te saklananlar arasında sayılır (sabit kısım); aktivasyon kısmı mikro-batch ile doğrusal küçülür.")
    R.test("tepe aktivasyon mikro-batch ile düşüyor", mem)
