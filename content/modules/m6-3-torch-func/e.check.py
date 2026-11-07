import copy
import torch
import torch.nn.functional as F
from torch.func import functional_call, vmap, grad
from labcheck import close, need, ensure, bench, fmt_time, uses_any


def ref_step(model, x, y, C, lr):
    B = x.shape[0]
    grads = []
    for i in range(B):
        model.zero_grad()
        F.cross_entropy(model(x[i:i + 1]), y[i:i + 1]).backward()
        g = [p.grad.clone() for p in model.parameters()]
        n = torch.sqrt(sum(t.pow(2).sum() for t in g))
        grads.append([t * min(1.0, C / (n.item() + 1e-12)) for t in g])
    with torch.no_grad():
        for j, p in enumerate(model.parameters()):
            p -= lr * sum(g[j] for g in grads) / B


def check(ns, R):
    src = ns["__source__"]
    R.test("vmap kullanılıyor", lambda: ensure(uses_any(src, ["vmap"]), "örnek başı gradyan için vmap kullan"))
    torch.manual_seed(0)
    base = torch.nn.Sequential(torch.nn.Linear(10, 16), torch.nn.Tanh(), torch.nn.Linear(16, 3)).double()
    x = torch.randn(12, 10, dtype=torch.float64); y = torch.randint(0, 3, (12,))

    def noiseless():
        a, b = copy.deepcopy(base), copy.deepcopy(base)
        need(ns, "dp_sgd_step")(a, x, y, max_norm=0.3, noise_mult=0.0, lr=0.1)
        ref_step(b, x, y, 0.3, 0.1)
        for p, q in zip(a.parameters(), b.parameters()):
            close(p, q)
    R.test("gürültüsüz adım = örnek başı kırpmalı döngü", noiseless)

    def noise():
        a, b = copy.deepcopy(base), copy.deepcopy(base)
        need(ns, "dp_sgd_step")(a, x, y, max_norm=0.3, noise_mult=0.0, lr=0.1)
        g = torch.Generator().manual_seed(0)
        ns["dp_sgd_step"](b, x, y, max_norm=0.3, noise_mult=2.0, lr=0.1, generator=g)
        diff = torch.cat([(p - q).flatten() for p, q in zip(a.parameters(), b.parameters())])
        want = 0.1 * 2.0 * 0.3 / 12
        ensure(abs(diff.std().item() - want) < 0.25 * want, f"gürültünün std'si {diff.std().item():.2e}, beklenen ≈ {want:.2e}")
    R.test("gürültü ölçeği σ·C·lr/B", noise)

    def clip():
        a = copy.deepcopy(base)
        before = [p.clone() for p in a.parameters()]
        need(ns, "dp_sgd_step")(a, x * 100, y, max_norm=0.01, noise_mult=0.0, lr=1.0)
        delta = torch.sqrt(sum((p - q).pow(2).sum() for p, q in zip(a.parameters(), before)))
        ensure(delta <= 0.01 + 1e-9, f"tüm örnekler kırpılınca adım ≤ C·lr olmalı ({delta.item():.4f})")
    R.test("kırpma: adım normu ≤ C·lr", clip)
    if "dp_sgd_step" not in ns:
        return
    m = torch.nn.Sequential(torch.nn.Linear(64, 256), torch.nn.ReLU(), torch.nn.Linear(256, 10))
    xb, yb = torch.randn(128, 64), torch.randint(0, 10, (128,))
    t_loop = bench(lambda: ref_step(m, xb, yb, 1.0, 0.0), repeat=1, number=1)
    t_vm = bench(lambda: ns["dp_sgd_step"](m, xb, yb, 1.0, 0.0, 0.0), repeat=3)
    R.table("DP-SGD adımı, B = 128", ["yöntem", "süre"], [["örnek başı backward döngüsü", fmt_time(t_loop)], ["senin vmap(grad)", fmt_time(t_vm)]])
