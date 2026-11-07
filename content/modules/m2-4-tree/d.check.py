import torch
from torch import nn
from labcheck import need, ensure


def check(ns, R):
    g = torch.Generator().manual_seed(0)
    X = torch.rand(400, 5, generator=g)
    y = (X[:, 3] > 0.35).long()

    def shape():
        torch.manual_seed(0)
        m = need(ns, "SoftStump", "sınıf")(5, 3)
        ensure(isinstance(m, nn.Module), "nn.Module olmalı")
        out = m(X)
        ensure(tuple(out.shape) == (400, 3), f"çıktı shape {tuple(out.shape)}")
        ensure(sum(p.numel() for p in m.parameters()) == 5 + 1 + 3 + 3, "parametreler: kapı (d + 1) + iki yaprak (C + C)")
    R.test("SoftStump: shape ve parametre sayısı", shape)

    def tr():
        torch.manual_seed(0)
        m = need(ns, "train_soft_stump")(X, y, 2)
        acc = (m(X).argmax(1) == y).float().mean().item()
        ensure(acc >= 0.95, f"doğruluk {acc:.2f}")
        lin = [mod for mod in m.modules() if isinstance(mod, nn.Linear)]
        ensure(lin, "kapı bir nn.Linear olmalı")
        w = lin[0].weight.detach().view(-1).abs()
        ensure(int(w.argmax()) == 3, f"kapı ağırlıkları {w.tolist()}; özellik 3'e yoğunlaşmalı")
        b = float(lin[0].bias.detach()); w3 = float(lin[0].weight.detach().view(-1)[3])
        return f"doğruluk {acc:.3f}; öğrenilen eşik ≈ {-b / w3:.3f} (gerçek 0.35)"
    R.test("eğitim: doğruluk ≥ 0.95, kapı doğru özelliği seçiyor", tr)
