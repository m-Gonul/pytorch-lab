import torch
from labcheck import close, need, ensure, bench, fmt_time


def check(ns, R_):
    def ev():
        idx, prob, R = need(ns, "random_sparse_mdp")(300, 3, 6, 1)
        pi = torch.zeros(300, dtype=torch.long)
        ar = torch.arange(300)
        dense = torch.zeros(300, 300, dtype=torch.float64)
        dense.index_put_((ar[:, None].expand(-1, 6), idx[0]), prob[0], accumulate=True)
        r = R[:, 0]
        a = need(ns, "eval_direct")(dense, r, 0.95)
        b, n = need(ns, "eval_iterative")(idx[0], prob[0], r, 0.95)
        close(b.numpy(), a.numpy(), atol=1e-7, what="yinelemeli = doğrudan")
        return f"{n} tarama"
    R_.test("eval_direct ve eval_iterative", ev)

    def mp():
        idx, prob, R = ns["random_sparse_mdp"](200, 4, 5, 2)
        V5, p5, _ = need(ns, "mpi_sparse")(idx, prob, R, 0.95, 5, 1e-10)
        V1, p1, _ = ns["mpi_sparse"](idx, prob, R, 0.95, 1, 1e-10)
        close(V5.numpy(), V1.numpy(), atol=1e-7, what="MPI k = 5 ile VI aynı V*")
    R_.test("mpi_sparse", mp)
    if "eval_direct" not in ns:
        return
    rows = []
    for S in (500, 1500, 3000):
        idx, prob, R = ns["random_sparse_mdp"](S, 1, 8, 3)
        ar = torch.arange(S)
        dense = torch.zeros(S, S, dtype=torch.float64)
        dense.index_put_((ar[:, None].expand(-1, 8), idx[0]), prob[0], accumulate=True)
        t1 = bench(lambda: ns["eval_direct"](dense, R[:, 0], 0.95), repeat=1, number=1)
        t2 = bench(lambda: ns["eval_iterative"](idx[0], prob[0], R[:, 0], 0.95, 1e-8), repeat=1, number=1)
        rows.append([S, fmt_time(t1), fmt_time(t2)])
    R_.table("Politika değerlendirme (k = 8 seyrek geçiş)", ["S", "doğrudan çözüm O(S³)", "seyrek yineleme"], rows,
             note="Büyük S'de doğrudan çözüm kübik büyür; seyrek yineleme tarama başına O(S·k). MPI bu yüzden pratikte tercih edilir.")
