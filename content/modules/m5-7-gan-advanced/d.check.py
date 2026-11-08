import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from labcheck import need, ensure


def check(ns, R_):
    X, y = need(ns, "digits16")()
    torch.manual_seed(0)
    clf = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                        nn.Flatten(), nn.Linear(1024, 64), nn.ReLU(), nn.Linear(64, 10))
    o = torch.optim.Adam(clf.parameters(), 1e-3)
    for _ in range(15):
        for i in range(0, len(X), 64):
            p = torch.randperm(len(X))[:64]
            loss = F.cross_entropy(clf(X[p]), y[p])
            o.zero_grad()
            loss.backward()
            o.step()
    clf.eval()

    def feat(x):
        with torch.no_grad():
            return clf[:-1](x).numpy()

    def t_arch():
        G, D = need(ns, "Generator")(32, 64), need(ns, "Discriminator")(64)
        x = G(torch.randn(5, 32))
        ensure(tuple(x.shape) == (5, 1, 16, 16) and float(x.abs().max()) <= 1, "üretici çıkışı (B, 1, 16, 16), tanh")
        ensure(tuple(D(x).shape) == (5,), "ayırıcı logit")
        ensure(not any(isinstance(m, nn.MaxPool2d) for m in list(G.modules()) + list(D.modules())), "havuz yok (DCGAN)")
        ensure(not isinstance(list(D.net)[1], nn.BatchNorm2d), "ayırıcı girişinde BN yok")
    R_.test("DCGAN mimarisi", t_arch)

    def t_train():
        fr = feat(X)
        torch.manual_seed(0)
        G, D = ns["Generator"](32, 64), ns["Discriminator"](64)
        t0 = time.perf_counter()
        G0 = ns["Generator"](32, 64)
        need(ns, "train_gan")(G, D, X, 30)
        G.eval()
        G0.eval()
        with torch.no_grad():
            xs, x0 = G(torch.randn(1000, 32)), G0(torch.randn(1000, 32))
        rows = []
        fd = need(ns, "fid_np")
        sets = [("gerçek (yarı) vs gerçek (yarı)", feat(X[::2]), None), ("DCGAN, 30 epoch", feat(xs), xs), ("eğitilmemiş üretici", feat(x0), x0),
                ("gerçek, yalnız 0–2 rakamları", feat(X[y <= 2]), X[y <= 2]), ("gerçek + N(0, 0.5²) gürültü", feat((X + 0.5 * torch.randn(X.shape)).clamp(-1, 1)), None)]
        res = {}
        for name, f, imgs in sets:
            ref = feat(X[1::2]) if name.startswith("gerçek (yarı)") else fr
            res[name] = fd(ref, f)
            cov = "—"
            if imgs is not None:
                with torch.no_grad():
                    c = torch.bincount(clf(imgs).argmax(1), minlength=10)
                cov = f"{int((c > 0.02 * len(imgs)).sum())}/10"
            rows.append([name, f"{res[name]:.1f}", cov])
        ensure(res["DCGAN, 30 epoch"] < 0.5 * res["eğitilmemiş üretici"] and res["DCGAN, 30 epoch"] > res["gerçek (yarı) vs gerçek (yarı)"], f"FID {res}")
        ensure(res["gerçek, yalnız 0–2 rakamları"] > res["gerçek (yarı) vs gerçek (yarı)"] * 3, "mod düşürme FID'i büyütür")
        R_.table(f"16 × 16 rakamlar: CNN sınıflandırıcı öznitelikleriyle FID ({time.perf_counter() - t0:.0f} sn eğitim)", ["örnekler", "FID", "sınıf kapsamı"], rows,
                 note="Gerçek veriyi ikiye bölmek bile sıfırdan uzak bir FID verir (sonlu örneklem yanlılığı + 900 örnek) — bu, ulaşılabilecek alt sınırdır. Eğitilmiş DCGAN tüm sınıfları kapsar ve FID'i rastgele üreticinin çok altına indirir; gerçek ama yalnız üç sınıflık görüntüler (mükemmel kalite, düşük çeşitlilik) FID'de kötüdür: FID çeşitliliği de ölçer.")
    R_.test("eğitim ve FID", t_train)
