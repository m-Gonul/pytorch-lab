import time
import torch
import torch.nn as nn
from labcheck import close, need, ensure


def _cnn():
    return nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Flatten(), nn.Linear(64 * 16, 64), nn.ReLU(), nn.Linear(64, 10))


def check(ns, R_):
    X, y = need(ns, "digits16")()
    g = torch.Generator().manual_seed(0)
    p = torch.randperm(len(X), generator=g)
    tr, te = p[:1200], p[1200:]

    def t_pe():
        torch.manual_seed(0)
        pe = need(ns, "PatchEmbed")(4, 3, 8)
        x = torch.randn(2, 3, 16, 12)
        a = pe(x)
        ensure(tuple(a.shape) == (2, 12, 8), "(B, N, D)")
        close(a.detach().numpy(), pe.forward_unfold(x).detach().numpy(), atol=1e-5, what="conv = unfold + doğrusal")
        v = need(ns, "ViT")()
        ensure(tuple(v(X[:3]).shape) == (3, 10), "çıkış")
        ensure(v.pos.shape[1] == 17 and v.cls.shape[-1] == 64, "17 token (16 yama + [class])")
    R_.test("yama embedding'i ve model", t_pe)

    def t_train():
        rows, acc = [], {}
        t00 = time.perf_counter()
        for name, mk, n, ep in (("CNN", _cnn, 300, 40), ("ViT ([class])", lambda: ns["ViT"](), 300, 40), ("ViT, konum embedding'i yok", lambda: ns["ViT"](pos=False), 300, 40),
                                ("CNN", _cnn, 1200, 20), ("ViT ([class])", lambda: ns["ViT"](), 1200, 20), ("ViT (GAP)", lambda: ns["ViT"](pool="gap"), 1200, 20)):
            torch.manual_seed(0)
            m = mk()
            a = need(ns, "train_eval")(m, X[tr[:n]], y[tr[:n]], X[te], y[te], ep)
            acc[(name, n)] = a
            rows.append([name, n, sum(q.numel() for q in m.parameters()), f"{a:.3f}"])
        ensure(acc[("CNN", 300)] > acc[("ViT ([class])", 300)], "az veride CNN önde")
        ensure(acc[("ViT ([class])", 300)] > acc[("ViT, konum embedding'i yok", 300)] + 0.08, "konum embedding'i gerekli")
        ensure(acc[("ViT ([class])", 1200)] > acc[("ViT ([class])", 300)] + 0.03, "ViT veriyle iyileşir")
        R_.table(f"16 × 16 rakamlar, P = 4 ({time.perf_counter() - t00:.0f} sn)", ["model", "eğitim örneği", "parametre", "test doğruluğu"], rows,
                 note="Aynı ölçekte CNN'in yerellik ve kaydırma eşdeğerliği önseli az veride belirgin üstünlük sağlar; veri arttıkça fark kapanır (makalenin ana bulgusunun minyatürü). Konum embedding'i olmadan ViT yamaların nerede olduğunu bilmez — bir 'yama torbası' (x2). [class] token'ı ile GAP benzer sonuç verir (makale Ek D.3; GAP farklı öğrenme oranı ister).")
    R_.test("CNN vs ViT", t_train)
