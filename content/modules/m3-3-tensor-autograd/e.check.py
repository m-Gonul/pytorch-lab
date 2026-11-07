import torch
from torch.func import jacrev, jacfwd
from labcheck import close, max_loops, need, ensure, bench, fmt_time


def check(ns, R):
    src = ns["__source__"]
    R.test("döngü yok", lambda: (max_loops(src, 0, func="jacobian_rev"), max_loops(src, 0, func="jacobian_fwd")))
    g = torch.Generator().manual_seed(0)
    A = torch.randn(6, 5, generator=g, dtype=torch.float64)
    f = lambda x: torch.tanh(A @ x) * x.sum()
    x = torch.randn(5, generator=g, dtype=torch.float64)
    R.test("jacobian_rev = jacrev", lambda: close(need(ns, "jacobian_rev")(f, x), jacrev(f)(x)))
    R.test("jacobian_fwd = jacfwd", lambda: close(need(ns, "jacobian_fwd")(f, x), jacfwd(f)(x)))
    if "jacobian_rev" not in ns or "jacobian_fwd" not in ns:
        return
    rows = []
    for n, m in ((8, 1024), (1024, 8)):
        W1 = torch.randn(256, n, generator=g) / n ** 0.5
        W2 = torch.randn(m, 256, generator=g) / 16
        h = lambda x: W2 @ torch.tanh(W1 @ x)
        xx = torch.randn(n, generator=g)
        tr = bench(lambda: ns["jacobian_rev"](h, xx), repeat=3)
        tf = bench(lambda: ns["jacobian_fwd"](h, xx), repeat=3)
        rows.append([f"n = {n}, m = {m}", fmt_time(tr), fmt_time(tf), "forward" if tf < tr else "reverse"])
    R.table("Tam Jacobian: reverse (m VJP) vs forward (n JVP)", ["boyutlar", "reverse", "forward", "kazanan"], rows,
            note="Kayıp fonksiyonu m = 1: tek VJP. Backprop'un reverse-mode olmasının nedeni bu.")
