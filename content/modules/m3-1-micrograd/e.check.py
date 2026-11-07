import torch
from labcheck import close, no_loops, need, ensure, bench, fmt_time


def check(ns, R):
    src = ns["__source__"]
    R.test("döngü yok", lambda: no_loops(src))
    g = torch.Generator().manual_seed(0)
    X, w, b = torch.randn(64, 5, generator=g, dtype=torch.float64), torch.randn(5, generator=g, dtype=torch.float64), 0.2

    def ps():
        G = need(ns, "per_sample_grads")(X, w, b)
        ensure(tuple(G.shape) == (64, 5), f"shape {tuple(G.shape)}, (64, 5) bekleniyordu")
        o = torch.tanh(X @ w + b)
        close(G, (1 - o ** 2)[:, None] * X, what="gᵢ = (1 − oᵢ²) xᵢ")
        close(G.sum(0), X.T @ (1 - o ** 2), what="satır toplamı = batch gradyanı")
        return "satırlar toplanınca c aşamasındaki X.T @ (1 − o²) çıkıyor"
    R.test("örnek başına gradyan doğru, toplamı batch gradyanı", ps)
    if "per_sample_grads" not in ns:
        return
    Xb, wb = torch.randn(1024, 32), torch.randn(32)

    def loop():
        out = []
        for x in Xb:
            ww = wb.clone().requires_grad_(True)
            torch.tanh(x @ ww + 0.1).backward()
            out.append(ww.grad)
        return torch.stack(out)
    t_loop = bench(loop, repeat=3, number=1)
    t_vm = bench(lambda: ns["per_sample_grads"](Xb, wb, 0.1), repeat=3)
    R.table("1024 örnek, d = 32: örnek başına gradyan", ["yöntem", "süre", "döngüye göre"],
            [["for + backward (1024 grafik)", fmt_time(t_loop), "1×"],
             ["senin per_sample_grads (vmap)", fmt_time(t_vm), f"{t_loop / t_vm:,.0f}× hızlı"]],
            note="Skaler micrograd ile aynı hesap 1024 × 32 Value düğümü demek; vmap birkaç tensör işlemine indirir.")
