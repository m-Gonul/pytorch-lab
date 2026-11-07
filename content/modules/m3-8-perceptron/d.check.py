import math
import torch
import torch.nn as nn
from labcheck import close, need, ensure


def rings(n, seed):
    g = torch.Generator().manual_seed(seed)
    r = torch.where(torch.rand(n, generator=g) < 0.5, 0.4, 1.0)
    th = torch.rand(n, generator=g) * 2 * math.pi
    X = torch.stack([r * torch.cos(th), r * torch.sin(th)], 1) + 0.08 * torch.randn(n, 2, generator=g)
    return X, (r > 0.5).long()


def check(ns, R_):
    def data():
        for fn, di, do in (("f1", 1, 1), ("f2", 2, 1), ("f3", 3, 2)):
            X, Y = need(ns, "make_data")(fn, 50, 0)
            ensure(X.shape == (50, di) and Y.shape == (50, do), f"{fn} şekilleri")
        m = need(ns, "mlp")(3, 2, 3, 16)
        ensure(sum(isinstance(l, nn.Linear) for l in m) == 4, "3 gizli + çıkış")
    R_.test("veri ve mimari", data)

    def depth():
        res = need(ns, "depth_table")(["f1", "f2", "f3"], [1, 2, 3, 4])
        rows = [[fn] + [f"{res[(fn, d)]:.2e}" for d in (1, 2, 3, 4)] for fn in ("f1", "f2", "f3")]
        for fn in ("f1", "f2", "f3"):
            ensure(min(res[(fn, d)] for d in (2, 3, 4)) < res[(fn, 1)], f"{fn}: derinlik yardımcı olmalı")
        R_.table("Test MSE: gizli katman sayısı (genişlik 64, Adam 2000 adım)", ["fonksiyon", "1", "2", "3", "4"], rows,
                 note="Aynı genişlikte bir katman çoğu zaman ilk iyileşmeyi yakalar ama keskin yapılar (f2'nin dar Gauss tepesi, f1'in salınımları) için derinlik çok daha verimli. Sonuçlar tohum ve optimizasyona da bağlı: farklar e aşamasında birçok tohumla ölçülüyor.")
    R_.test("ders alıştırması 1: derinlik", depth)

    def sep():
        X, y = rings(600, 0)
        torch.manual_seed(0)
        m = need(ns, "mlp")(2, 1, 3, 32)
        opt = torch.optim.Adam(m.parameters(), lr=1e-2)
        for _ in range(800):
            loss = nn.functional.binary_cross_entropy_with_logits(m(X)[:, 0], y.float())
            opt.zero_grad()
            loss.backward()
            opt.step()
        acc_in = need(ns, "penultimate_separability")(m, X, y)
        lin = nn.Linear(2, 1)
        optl = torch.optim.Adam(lin.parameters(), lr=0.05)
        for _ in range(500):
            l = nn.functional.binary_cross_entropy_with_logits(lin(X)[:, 0], y.float())
            optl.zero_grad()
            l.backward()
            optl.step()
        acc_raw = float(((lin(X)[:, 0] > 0).long() == y).float().mean())
        ensure(acc_in > 0.95 and acc_raw < 0.75, "ham uzayda doğrusal ayrılamaz, son gizli katmanda ayrılabilir")
        R_.table("Halkalar (ders alıştırması 3)", ["girdi", "doğrusal sınıflandırıcı doğruluğu"], [["ham (x₁, x₂)", f"{acc_raw:.2f}"], ["MLP'nin son gizli katmanı (32 boyut)", f"{acc_in:.2f}"]],
                 note="MLP'nin gizli katmanları, x1'deki elle tasarlanmış (x₁, x₂, x₁² + x₂²) dönüşümünün işini öğrenir.")
    R_.test("öğrenilmiş öznitelikler", sep)
