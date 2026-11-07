import copy
import torch
from torch import nn
from labcheck import close, need, ensure


def check(ns, R):
    def eq():
        Cls = need(ns, "MyAdamW", "sınıf")
        ensure(issubclass(Cls, torch.optim.Optimizer), "torch.optim.Optimizer'dan türemeli")
        torch.manual_seed(0)
        m1 = nn.Sequential(nn.Linear(4, 16), nn.Tanh(), nn.Linear(16, 3)).double()
        m2 = copy.deepcopy(m1)
        groups = lambda m: [{"params": [m[0].weight, m[2].weight], "weight_decay": 0.05}, {"params": [m[0].bias, m[2].bias], "weight_decay": 0.0}]
        o1 = Cls(groups(m1), lr=0.01, betas=(0.85, 0.99), eps=1e-7)
        o2 = torch.optim.AdamW(groups(m2), lr=0.01, betas=(0.85, 0.99), eps=1e-7)
        X = torch.randn(32, 4, dtype=torch.float64); y = torch.randint(0, 3, (32,))
        for _ in range(30):
            for m, o in ((m1, o1), (m2, o2)):
                o.zero_grad()
                nn.functional.cross_entropy(m(X), y).backward()
                o.step()
        for (n, a), (_, b) in zip(m1.named_parameters(), m2.named_parameters()):
            close(a, b, rtol=1e-12, atol=1e-14, what=n)
        st = o1.state[m1[0].weight]
        ensure(int(st["step"]) == 30, "state['step'] 30 olmalı")
        return "30 adım, 2 param_group: torch.optim.AdamW ile 12 basamak aynı"
    R.test("MyAdamW = torch.optim.AdamW", eq)

    def clo():
        Cls = need(ns, "MyAdamW", "sınıf")
        w = torch.zeros(2, requires_grad=True)
        o = Cls([w], lr=0.1, weight_decay=0.0)
        def closure():
            o.zero_grad(); L = ((w - 3) ** 2).sum(); L.backward(); return L
        L = o.step(closure)
        ensure(L is not None and abs(float(L.detach()) - 18) < 1e-9, "step(closure) kaybı döndürmeli")
        close(w.detach(), [0.1, 0.1], atol=1e-6, what="ilk adım ≈ lr")
    R.test("step(closure) destekleniyor", clo)
