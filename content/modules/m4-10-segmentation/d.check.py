import time
import torch
import torch.nn.functional as F
from labcheck import close, need, ensure


def check(ns, R_):
    def t_adj():
        torch.manual_seed(0)
        for s, p, H in ((1, 0, 6), (2, 1, 8), (2, 0, 9), (3, 1, 10)):
            w = torch.randn(4, 3, 3, 3)
            Ho = (H + 2 * p - 3) // s + 1
            y = torch.randn(2, 4, Ho, Ho)
            g = need(ns, "conv_input_grad")(y, w, s, p, H)
            op = H - ((Ho - 1) * s - 2 * p + 3)
            close(g.numpy(), F.conv_transpose2d(y, w, stride=s, padding=p, output_padding=op).numpy(), atol=1e-5, what=f"∂conv/∂x = conv_transpose (s={s}, p={p})")
        t = need(ns, "bilinear_init_")(torch.nn.ConvTranspose2d(2, 2, 4, 2, 1))
        x = torch.rand(1, 2, 6, 6)
        close(t(x)[:, :, 1:-1, 1:-1].detach().numpy(), F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)[:, :, 1:-1, 1:-1].numpy(), atol=1e-5, what="bilineer başlatılmış tconv = F.interpolate (iç bölge)")
    R_.test("transposed conv = adjoint; bilineer başlatma", t_adj)

    def t_train():
        X, Y = need(ns, "seg_dataset")(1000, 0)
        Xt, Yt = ns["seg_dataset"](200, 1)
        rows, res = [], {}
        t00 = time.perf_counter()
        for name, mk in (("FCN adım 8, rastgele tconv", lambda: ns["FCN"](4, 12, False, False)), ("FCN adım 8, bilineer tconv", lambda: ns["FCN"](4, 12, False, True)),
                         ("FCN-8s düzeni (atlamalı)", lambda: ns["FCN"](4, 12, True, True)), ("U-Net", lambda: need(ns, "UNet")(4, 12))):
            torch.manual_seed(0)
            m = mk()
            ensure(tuple(m(X[:2]).shape) == (2, 4, 64, 64), f"{name}: çıkış (N, C, 64, 64)")
            opt = torch.optim.Adam(m.parameters(), 3e-3)
            for ep in range(5):
                m.train()
                perm = torch.randperm(len(X))
                for i in range(0, len(X), 32):
                    idx = perm[i:i + 32]
                    loss = F.cross_entropy(m(X[idx]), Y[idx])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            m.eval()
            with torch.no_grad():
                P = m(Xt).argmax(1)
            res[name] = (need(ns, "miou")(P, Yt, 4), need(ns, "edge_acc")(P, Yt, 2))
            rows.append([name, sum(q.numel() for q in m.parameters()), f"{float((P == Yt).float().mean()):.3f}", f"{res[name][0]:.3f}", f"{res[name][1]:.3f}"])
        r = list(res.values())
        ensure(r[1][1] > r[0][1] and r[2][1] > r[1][1] and r[3][1] > r[1][1] + 0.05, "kenar doğruluğu: rastgele < bilineer < atlamalı / U-Net")
        ensure(r[3][0] > r[1][0], "U-Net mIoU > tek yukarı örneklemeli FCN")
        R_.table(f"64 × 64 sentetik sahneler, 1000 görüntü × 5 epoch ({time.perf_counter() - t00:.0f} sn)", ["model", "parametre", "piksel doğr.", "mIoU", "kenar doğr. (±2 px)"], rows,
                 note="Piksel doğruluğu arka planın çokluğu yüzünden yanıltıcı derecede yüksek; mIoU ve özellikle kenar doğruluğu farkı gösterir. Tek ×8 yukarı örnekleme sınırları bulanıklaştırır; sığ katmanlardan gelen bilgi (FCN-8s toplama, U-Net birleştirme) kenarları geri kazandırır. Bilineer başlatma, rastgele başlatılmış büyük tconv'un öğrenmesi gereken interpolasyonu hazır verir.")
    R_.test("FCN vs U-Net", t_train)
