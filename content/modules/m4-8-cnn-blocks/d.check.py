import torch
import torch.nn as nn
from labcheck import close, need, ensure


def check(ns, R_):
    def mods():
        x = torch.randn(2, 32, 9, 9)
        ds = need(ns, "DepthwiseSeparable")(32, 64)
        ensure(tuple(ds(x).shape) == (2, 64, 9, 9), "şekil")
        ensure(sum(p.numel() for p in ds.parameters()) == 9 * 32 + 32 * 64 + 64, "parametre: 9C + C·C' + C'")
        se = need(ns, "SE")(32, 8)
        y = se(x)
        r = y / x
        ensure(tuple(y.shape) == x.shape and torch.allclose(r, r[:, :, :1, :1].expand_as(r), atol=1e-5), "SE kanal ölçeklemesi")
        inc = need(ns, "Inception")(192, 64, 96, 128, 16, 32, 32)
        ensure(tuple(inc(torch.randn(1, 192, 28, 28)).shape) == (1, 256, 28, 28), "Inception 3a: 256 kanal")
        lrn = nn.LocalResponseNorm(5, alpha=5e-4, beta=0.75, k=2.0)
        a = torch.randn(1, 7, 3, 3) * 5
        ref = torch.stack([a[0, i] / (2 + 1e-4 * (a[0, max(0, i - 2):i + 3] ** 2).sum(0)) ** 0.75 for i in range(7)])
        close(lrn(a)[0].numpy(), ref.numpy(), rtol=1e-5, what="nn.LocalResponseNorm(α/n kuralı) = AlexNet LRN (α = 5·10⁻⁴/5)")
    R_.test("bloklar", mods)

    def ex5():
        M = need(ns, "ex5_models")()
        cnt = {k: sum(p.numel() for p in m.parameters() if p.requires_grad) for k, m in M.items()}
        ensure(cnt["cnn2"] == 61514 and cnt["fc3"] == 62350 and cnt["fc1"] == 266610 and cnt["cnn4"] == 1546954, "parametre sayıları")
        X, y = need(ns, "shifted_digits")(3, 0)
        Xt, yt = ns["shifted_digits"](1, 1)
        ensure(tuple(X.shape) == (5391, 1, 28, 28) and float(X.sum(dim=(1, 2, 3)).min()) > 0, "kaydırılmış rakamlar")
        acc = {k: need(ns, "train_eval")(M[k], X, y, Xt, yt, 8) for k in ("fc1", "cnn2", "fc3")}
        ensure(acc["cnn2"] > acc["fc3"] + 0.05 and acc["cnn2"] > acc["fc1"], "eşit bütçede CNN, 4× büyük FC'yi de geçmeli")
        R_.table("Kaydırılmış rakamlar (28×28), 8 epoch", ["model", "parametre", "test doğruluğu"], [[k, cnt[k], f"{acc[k]:.3f}"] for k in acc],
                 note="Rakam tuvalin her yerinde olabilir: FC her konum için desenleri ayrı öğrenmeli; CNN aynı filtreyi her yerde uygular (eşdeğerlik) ve havuzlama küçük kaymaları emer. Model 4 (BN'li büyük CNN) ödev ölçeğinde: daha uzun eğitimle en iyisi ama 25× parametre.")
    R_.test("ders alıştırması 5", ex5)
