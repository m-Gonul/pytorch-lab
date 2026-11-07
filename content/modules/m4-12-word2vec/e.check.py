import time
import torch
from labcheck import close, need, ensure, fmt_time


def check(ns, R_):
    def t():
        torch.manual_seed(0)
        Win, Wout = torch.randn(50, 8), torch.randn(50, 8)
        c, o = torch.randint(0, 50, (16,)), torch.randint(0, 50, (16,))
        s = Win[c] @ Wout.T
        close(need(ns, "full_softmax_loss")(Win, Wout, c, o).item(), (torch.logsumexp(s, 1) - s[torch.arange(16), o]).mean().item(), rtol=1e-5, what="tam softmax")
        neg = torch.randint(0, 50, (16, 5))
        ref = -(torch.log(torch.sigmoid((Win[c] * Wout[o]).sum(1))) + torch.log(torch.sigmoid(-(Win[c][:, None] * Wout[neg]).sum(-1))).sum(1)).mean()
        close(need(ns, "neg_sampling_loss")(Win, Wout, c, o, neg).item(), ref.item(), rtol=1e-5, what="negatif örnekleme")
        rows = []
        for V in (1000, 10000, 50000):
            Wi = torch.randn(V, 100, requires_grad=True)
            Wo = torch.randn(V, 100, requires_grad=True)
            c, o = torch.randint(0, V, (1024,)), torch.randint(0, V, (1024,))
            neg = torch.randint(0, V, (1024, 5))
            r = [V]
            for f, args in ((ns["full_softmax_loss"], (Wi, Wo, c, o)), (ns["neg_sampling_loss"], (Wi, Wo, c, o, neg))):
                best = 1e9
                for _ in range(2):
                    t0 = time.perf_counter()
                    f(*args).backward()
                    best = min(best, time.perf_counter() - t0)
                r.append(fmt_time(best))
            rows.append(r)
        R_.table("İleri + geri, batch 1024, d = 100 (CPU)", ["V", "tam softmax", "negatif örnekleme (k = 5)"], rows,
                 note="Tam softmax batch başına (B × V) logit üretir: V ile doğrusal; negatif örnekleme yalnız B × (k + 1) iç çarpım. Gerçek sözlüklerde (V ~ 10⁶) tam softmax pratik değildir; word2vec'in hızını negatif örnekleme ve hiyerarşik softmax (x2, O(log V)) sağlar. (Burada tam tabloların yoğun gradyanı da süreye katkıda bulunur.)")
    R_.test("maliyet", t)
