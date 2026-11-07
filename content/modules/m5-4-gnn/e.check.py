import torch
from torch import nn
from labcheck import close, max_loops, need, ensure, bench, fmt_time


def rand_graphs(g, k, d):
    out = []
    for _ in range(k):
        n = int(torch.randint(3, 25, (1,), generator=g))
        A = (torch.rand(n, n, generator=g) < 0.25).triu(1)
        A = A | A.T
        out.append((torch.randn(n, d, generator=g, dtype=torch.float64), A.nonzero().T))
    return out


def check(ns, R):
    src = ns["__source__"]
    R.test("collate: en fazla tek seviye döngü (graflar)", lambda: max_loops(src, 1, func="collate"))
    R.test("readout döngüsüz", lambda: max_loops(src, 0, func="readout"))
    g = torch.Generator().manual_seed(0)
    graphs = rand_graphs(g, 20, 4)
    prop = need(ns, "propagate")

    def col():
        X, EI, batch = need(ns, "collate")(graphs)
        ensure(X.shape[0] == sum(x.shape[0] for x, _ in graphs) and EI.shape[1] == sum(e.shape[1] for _, e in graphs), "toplam düğüm/kenar sayısı")
        ensure(torch.equal(batch[EI[0]], batch[EI[1]]), "hiçbir kenar iki grafı birbirine bağlamamalı")
        ensure(torch.equal(torch.bincount(batch), torch.tensor([x.shape[0] for x, _ in graphs])), "batch vektörü")
    R.test("collate: blok-köşegen graf", col)

    def ro():
        x = torch.tensor([[1.0, -1], [2, -2], [5, 0], [0, 3], [1, 1]], dtype=torch.float64)
        close(need(ns, "readout")(x, torch.tensor([0, 0, 1, 1, 1]), 2), [[1.5, -1.5, 2, -1], [2, 4 / 3, 5, 3]])
    R.test("readout: [ortalama, max] (negatif değerler dahil)", ro)

    W1 = torch.randn(4, 16, generator=g, dtype=torch.float64); W2 = torch.randn(16, 16, generator=g, dtype=torch.float64)

    def emb_one(x, ei):
        h = torch.relu(prop(x @ W1, ei, x.shape[0]))
        h = prop(h @ W2, ei, x.shape[0])
        return torch.cat([h.mean(0), h.max(0).values])

    def eq():
        X, EI, batch = need(ns, "collate")(graphs)
        h = torch.relu(prop(X @ W1, EI, X.shape[0]))
        h = prop(h @ W2, EI, X.shape[0])
        close(need(ns, "readout")(h, batch, len(graphs)), torch.stack([emb_one(x, e) for x, e in graphs]))
    R.test("birleşik graf = tek tek işlenen graflar", eq)
    if "readout" not in ns or "collate" not in ns:
        return
    big = rand_graphs(g, 500, 4)
    big = [(x.float(), e) for x, e in big]
    W1f, W2f = W1.float(), W2.float()

    def loop_all():
        outs = []
        for x, e in big:
            h = torch.relu(prop(x @ W1f, e, x.shape[0])); h = prop(h @ W2f, e, x.shape[0])
            outs.append(torch.cat([h.mean(0), h.max(0).values]))
        return torch.stack(outs)

    def batched():
        X, EI, b = ns["collate"](big)
        h = torch.relu(prop(X @ W1f, EI, X.shape[0])); h = prop(h @ W2f, EI, X.shape[0])
        return ns["readout"](h, b, len(big))
    t_loop = bench(loop_all, repeat=3, number=1)
    t_b = bench(batched, repeat=3)
    R.table("500 graf, 2 katman GCN + readout", ["yöntem", "süre"], [["graf başına döngü", fmt_time(t_loop)], ["collate + tek geçiş", fmt_time(t_b)]])
