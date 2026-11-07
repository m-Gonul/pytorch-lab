import torch
from labcheck import close, need, ensure


def check(ns, R):
    g = torch.Generator().manual_seed(0)
    T = torch.rand(4, 4, generator=g, dtype=torch.float64); T /= T.sum(0, keepdim=True)

    def sim():
        tr = need(ns, "simulate")(T, 1, 200, 500, 3)
        ensure(tuple(tr.shape) == (500, 201) and bool((tr[:, 0] == 1).all()), "şekil")
        Th = need(ns, "count_mle")(tr, 4)
        close(Th.sum(0).numpy(), torch.ones(4).numpy(), atol=1e-12, what="sütun toplamları 1")
        close(Th.numpy(), T.numpy(), atol=0.02, what="100 bin geçişten kestirim")
    R.test("simulate ve count_mle", sim)

    def gm():
        tr = ns["simulate"](T, 0, 100, 50, 5)
        a = ns["count_mle"](tr, 4)
        b = need(ns, "grad_mle")(tr, 4)
        close(b.numpy(), a.numpy(), atol=2e-3, what="gradyanla MLE = sayma MLE")
        rows = []
        for n in (10, 100, 1000):
            e = (ns["count_mle"](ns["simulate"](T, 0, 100, n, 7), 4) - T).abs().max().item()
            rows.append([n * 100, f"{e:.4f}"])
        R.table("Kestirim hatası (en büyük mutlak)", ["geçiş sayısı", "hata"], rows, note="Hata ≈ 1/√n ile azalır: 100 kat veri, 10 kat daha küçük hata.")
    R.test("grad_mle", gm)
