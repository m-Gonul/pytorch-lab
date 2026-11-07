import torch
import torch.nn.functional as F
from labcheck import no_loops, need, ensure, bench, fmt_time


def one_k(D, ytr, k, C):
    idx = torch.sort(D, dim=1, stable=True).indices[:, :k]
    return F.one_hot(ytr[idx], C).sum(1).argmax(1)


def check(ns, R):
    src = ns["__source__"]
    R.test("döngü yok", lambda: no_loops(src))
    R.test("elle örnek: k = 1, 2, 3, 4 → 1, 0, 0, 0",
           lambda: ensure(need(ns, "predict_many_k")(torch.tensor([[5.0, 1.0, 5.0, 13.0]]), torch.tensor([0, 1, 0, 1]), torch.tensor([1, 2, 3, 4]), 2).reshape(-1).tolist() == [1, 0, 0, 0],
                          "k=2'de oylar 1–1 berabere: küçük etiket (0) kazanmalı"))
    g = torch.Generator().manual_seed(4)
    Xtr, Xte = torch.randn(400, 6, generator=g), torch.randn(120, 6, generator=g)
    ytr = torch.randint(0, 5, (400,), generator=g)
    D = torch.cdist(Xte, Xtr) ** 2
    ks = torch.arange(1, 21)

    def eq():
        P = need(ns, "predict_many_k")(D, ytr, ks, 5)
        ensure(tuple(P.shape) == (20, 120), f"shape {tuple(P.shape)}, (20, 120) bekleniyordu")
        for i, k in enumerate(ks.tolist()):
            ensure(bool((P[i] == one_k(D, ytr, k, 5)).all()), f"k={k} tahminleri farklı")
        return "20 k değerinin hepsi ayrı ayrı hesaplananla aynı"
    R.test("20 k değeri, tek tek tahminle eşleşiyor", eq)
    if "predict_many_k" not in ns:
        return
    Xtr2, Xte2 = torch.randn(5000, 64, generator=g), torch.randn(500, 64, generator=g)
    ytr2 = torch.randint(0, 10, (5000,), generator=g)
    D2 = torch.cdist(Xte2, Xtr2) ** 2
    t_loop = bench(lambda: [one_k(D2, ytr2, k, 10) for k in range(1, 21)], repeat=3)
    t_vec = bench(lambda: ns["predict_many_k"](D2, ytr2, ks, 10), repeat=3)
    R.table("20 k değeri (500 test × 5000 eğitim)", ["yöntem", "süre", "döngüye göre"],
            [["for k: her seferinde sırala + oyla", fmt_time(t_loop), "1×"],
             ["senin predict_many_k (tek sıralama + cumsum)", fmt_time(t_vec), f"{t_loop / t_vec:,.1f}× hızlı"]])
    te, tr = Xte2[:200], Xtr2
    t_bc = bench(lambda: ((te[:, None] - tr[None]) ** 2).sum(-1), repeat=3)
    t_ex = bench(lambda: ((te ** 2).sum(1, keepdim=True) + (tr ** 2).sum(1) - 2 * te @ tr.T).clamp_min(0), repeat=3)
    t_cd = bench(lambda: torch.cdist(te, tr), repeat=3)
    R.table("Mesafe matrisi, 200 × 5000, d = 64", ["yöntem", "süre", "ara bellek"],
            [["broadcasting fark: (200, 5000, 64)", fmt_time(t_bc), f"{200 * 5000 * 64 * 4 / 2**20:,.0f} MB"],
             ["açılım: ‖a‖² + ‖b‖² − 2ab", fmt_time(t_ex), f"{200 * 5000 * 4 / 2**20:,.1f} MB"],
             ["torch.cdist", fmt_time(t_cd), "kütüphaneye bağlı"]],
            note="Açılım, işi tek bir matris çarpımına (BLAS) taşıdığı için hem hızlı hem bellek dostu.")
