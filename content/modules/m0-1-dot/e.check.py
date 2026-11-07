import torch
import torch.nn.functional as F
from labcheck import close, no_loops, need, bench, fmt_time


def check(ns, R):
    src = ns["__source__"]
    R.test("döngü yok", lambda: no_loops(src))
    g = torch.Generator().manual_seed(0)
    U, V = torch.randn(64, 16, generator=g), torch.randn(64, 16, generator=g)
    Q, K = torch.randn(10, 16, generator=g), torch.randn(30, 16, generator=g)
    R.test("rowwise_dot doğru", lambda: close(need(ns, "rowwise_dot")(U, V), (U * V).sum(1), rtol=1e-5, atol=1e-5))
    R.test("pairwise_cosine doğru, shape (n, m)", lambda: close(need(ns, "pairwise_cosine")(Q, K), F.cosine_similarity(Q[:, None], K[None], dim=-1), rtol=1e-4, atol=1e-5))
    if "rowwise_dot" not in ns:
        return
    rd = ns["rowwise_dot"]
    A, B = torch.randn(4096, 64), torch.randn(4096, 64)
    t_loop = bench(lambda: torch.stack([torch.dot(a, b) for a, b in zip(A, B)]), repeat=3)
    t_vec = bench(lambda: rd(A, B))
    t_mul = bench(lambda: (A * B).sum(1))
    R.table("4096 çiftin iç çarpımı (d = 64)", ["yöntem", "süre", "döngüye göre"],
            [["for + torch.dot", fmt_time(t_loop), "1×"],
             ["senin rowwise_dot", fmt_time(t_vec), f"{t_loop / t_vec:,.0f}× hızlı"],
             ["(A * B).sum(1)", fmt_time(t_mul), f"{t_loop / t_mul:,.0f}× hızlı"]],
            note="(A * B).sum(1) önce (4096, 64) ara tensörü yazar; einsum bunu atlayabilir. Farkı belleğe yazılan bayt sayısı belirler.")
