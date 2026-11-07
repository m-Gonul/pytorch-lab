import torch
from labcheck import close, no_loops, need, ensure, bench, fmt_time


def check(ns, R):
    src = ns["__source__"]
    R.test("döngü yok", lambda: no_loops(src))
    g = torch.Generator().manual_seed(0)
    W1, b1 = torch.randn(5, 32, generator=g, dtype=torch.float64), torch.randn(32, generator=g, dtype=torch.float64) * 0.1
    W2, b2 = torch.randn(32, 3, generator=g, dtype=torch.float64), torch.randn(3, generator=g, dtype=torch.float64) * 0.1
    X = torch.randn(20, 5, generator=g, dtype=torch.float64)
    masks = torch.rand(50, 20, 32, generator=g) >= 0.5

    def eq():
        out = need(ns, "mc_logits")(W1, b1, W2, b2, X, masks, 0.5)
        ensure(tuple(out.shape) == (50, 20, 3), f"shape {tuple(out.shape)}")
        H = torch.relu(X @ W1 + b1)
        ref = torch.stack([(H * masks[t] / 0.5) @ W2 + b2 for t in range(50)])
        close(out, ref)
    R.test("mc_logits = T geçiş döngüsü", eq)

    def ent():
        L = torch.randn(50, 20, 3, generator=g, dtype=torch.float64)
        P = L.softmax(-1).mean(0)
        close(need(ns, "predictive_entropy")(L), -(P * P.log()).sum(-1))
        u = need(ns, "predictive_entropy")(torch.zeros(4, 2, 3, dtype=torch.float64))
        close(u, torch.full((2,), float(torch.log(torch.tensor(3.0)))), what="eşit olasılık → log C")
    R.test("predictive_entropy", ent)
    if "mc_logits" not in ns:
        return
    W1f, b1f, W2f, b2f, Xf = W1.float(), b1.float(), W2.float(), b2.float(), torch.randn(512, 5, generator=g)
    mk = torch.rand(100, 512, 32, generator=g) >= 0.5
    H = torch.relu(Xf @ W1f + b1f)
    t_loop = bench(lambda: torch.stack([(H * mk[t] / 0.5) @ W2f + b2f for t in range(100)]), repeat=3)
    t_vec = bench(lambda: ns["mc_logits"](W1f, b1f, W2f, b2f, Xf, mk, 0.5), repeat=3)
    R.table("MC dropout, T = 100 geçiş, N = 512", ["yöntem", "süre", "döngüye göre"],
            [["for t: geçiş", fmt_time(t_loop), "1×"], ["senin mc_logits", fmt_time(t_vec), f"{t_loop / t_vec:,.1f}× hızlı"]])
