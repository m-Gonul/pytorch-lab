import torch
from torch import nn
from labcheck import close, max_loops, need, ensure, bench, fmt_time


def gp_loop(D, real, fake, alpha):
    vals = []
    for i in range(real.shape[0]):
        x = (alpha[i] * real[i] + (1 - alpha[i]) * fake[i]).detach().requires_grad_(True)
        g, = torch.autograd.grad(D(x[None]).sum(), x, create_graph=True)
        vals.append((g.norm() - 1) ** 2)
    return torch.stack(vals).mean()


def check(ns, R):
    src = ns["__source__"]
    R.test("gradient_penalty döngüsüz", lambda: max_loops(src, 0, func="gradient_penalty"))
    torch.manual_seed(0)
    D = nn.Sequential(nn.Linear(4, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1)).double()
    g = torch.Generator().manual_seed(0)
    real, fake = torch.randn(16, 4, generator=g, dtype=torch.float64), torch.randn(16, 4, generator=g, dtype=torch.float64)
    alpha = torch.rand(16, 1, generator=g, dtype=torch.float64)
    R.test("örnek başı döngüyle aynı", lambda: close(need(ns, "gradient_penalty")(D, real, fake, alpha), gp_loop(D, real, fake, alpha)))

    def lin():
        L = nn.Linear(4, 1, bias=False).double()
        with torch.no_grad():
            L.weight.copy_(torch.tensor([[3.0, 0, 4, 0]]))
        close(need(ns, "gradient_penalty")(L, real, fake, alpha), 16.0, what="‖w‖ = 5 → (5 − 1)² = 16")
    R.test("doğrusal critic: (‖w‖ − 1)²", lin)

    def to_params():
        D.zero_grad()
        need(ns, "gradient_penalty")(D, real, fake, alpha).backward()
        ensure(all(p.grad is not None and p.grad.abs().sum() > 0 for p in D.parameters() if p.dim() > 1), "ceza D'nin ağırlıklarına gradyan vermeli (create_graph=True)")
        z = lambda p: torch.zeros_like(p) if p.grad is None else p.grad.clone()
        a = [z(p) for p in D.parameters()]
        D.zero_grad(set_to_none=True); gp_loop(D, real, fake, alpha).backward()
        for x, y in zip(a, [z(p) for p in D.parameters()]):
            close(x, y, what="parametre gradyanı")
    R.test("ikinci türev: D parametrelerine gradyan", to_params)

    def bn():
        Dbn = nn.Sequential(nn.Linear(4, 16), nn.BatchNorm1d(16), nn.Tanh(), nn.Linear(16, 1)).double().train()
        a = need(ns, "gradient_penalty")(Dbn, real, fake, alpha).item()
        Dbn.eval(); b = ns["gradient_penalty"](Dbn, real, fake, alpha).item()
        return f"BatchNorm'lu critic: train modunda {a:.4f}, eval modunda {b:.4f}. Örnekler birbirine bağlanınca toplamın gradyanı örnek başı gradyan olmaz."
    R.test("not: BatchNorm örnekleri birbirine bağlar", bn)
    if "gradient_penalty" not in ns:
        return
    D32 = nn.Sequential(nn.Linear(64, 256), nn.Tanh(), nn.Linear(256, 1))
    r, f, a = torch.randn(256, 64), torch.randn(256, 64), torch.rand(256, 1)
    t_loop = bench(lambda: gp_loop(D32, r, f, a), repeat=3, number=1)
    t_vec = bench(lambda: ns["gradient_penalty"](D32, r, f, a), repeat=3)
    R.table("B = 256, d = 64", ["yöntem", "süre"], [["örnek başı autograd.grad", fmt_time(t_loop)], ["senin (tek çağrı)", fmt_time(t_vec)]])
