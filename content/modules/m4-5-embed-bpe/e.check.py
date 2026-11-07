import torch
import torch.nn.functional as F
from labcheck import close, no_loops, need, ensure, bench, fmt_time


def check(ns, R):
    src = ns["__source__"]
    R.test("döngü yok", lambda: no_loops(src))
    g = torch.Generator().manual_seed(0)
    W = torch.randn(50, 8, generator=g, dtype=torch.float64)

    def eq():
        ids = torch.randint(0, 50, (40,), generator=g)
        offsets = torch.tensor([0, 3, 3, 10, 25, 39])
        close(need(ns, "embedding_bag_mean")(W, ids, offsets), F.embedding_bag(ids, W, offsets, mode="mean"))
    R.test("F.embedding_bag(mode='mean') ile aynı (boş grup dahil)", eq)

    def gr():
        Wa = W.clone().requires_grad_(True); Wb = W.clone().requires_grad_(True)
        ids = torch.randint(0, 50, (30,), generator=g); off = torch.tensor([0, 7, 20])
        need(ns, "embedding_bag_mean")(Wa, ids, off).pow(2).sum().backward()
        F.embedding_bag(ids, Wb, off, mode="mean").pow(2).sum().backward()
        close(Wa.grad, Wb.grad)
    R.test("gradyan da aynı", gr)
    if "embedding_bag_mean" not in ns:
        return
    Wb = torch.randn(10000, 64, generator=g)
    lens = torch.randint(1, 50, (2000,), generator=g)
    off = torch.cat([torch.zeros(1, dtype=torch.long), lens.cumsum(0)[:-1]])
    ids = torch.randint(0, 10000, (int(lens.sum()),), generator=g)
    t_loop = bench(lambda: torch.stack([Wb[ids[o:o + l]].mean(0) for o, l in zip(off.tolist(), lens.tolist())]), repeat=3, number=1)
    t_vec = bench(lambda: ns["embedding_bag_mean"](Wb, ids, off), repeat=3)
    t_ref = bench(lambda: F.embedding_bag(ids, Wb, off, mode="mean"), repeat=3)
    R.table("2000 torba, ort. 25 token, d = 64", ["yöntem", "süre"],
            [["for torba: mean", fmt_time(t_loop)], ["senin embedding_bag_mean", fmt_time(t_vec)], ["F.embedding_bag (fused)", fmt_time(t_ref)]],
            note="Fused kernel ara (toplam id × d) tensörünü (weight[ids]) hiç oluşturmaz.")
