import torch
import torch.nn.functional as F
from labcheck import close, need, ensure


def check(ns, R_):
    def t():
        torch.manual_seed(0)
        M = need(ns, "MappingNet")(16, 32, 8)
        z = torch.randn(4, 16)
        close(M(z).detach().numpy(), M(3 * z).detach().numpy(), atol=1e-5, what="piksel normu: z ölçeğinden bağımsız")
        ensure(sum(isinstance(m, torch.nn.Linear) for m in M.modules()) == 8, "8 doğrusal katman")
        A = need(ns, "AdaIN")(6, 32)
        x = torch.randn(4, 6, 8, 8) * 5 + 3
        w = torch.randn(4, 32)
        y = A(x, w)
        ys, yb = A.style(w).chunk(2, 1)
        close(y.mean((2, 3)).detach().numpy(), yb.detach().numpy(), atol=1e-4, what="çıktı ortalaması = y_b")
        close(y.std((2, 3), unbiased=False).detach().numpy(), ys.abs().detach().numpy(), atol=1e-3, what="çıktı std = |y_s|")
        close(A(x * 7 - 2, w).detach().numpy(), y.detach().numpy(), atol=1e-4, what="girdinin önceki stili silinir")
        Nz = need(ns, "NoiseInjection")(6)
        ensure(torch.equal(Nz(x), x), "ölçek 0 başlar → başlangıçta etkisiz")
        with torch.no_grad():
            Nz.scale.fill_(0.5)
        n = torch.randn(4, 1, 8, 8)
        close(Nz(x, n).detach().numpy(), (x + 0.5 * n).numpy(), atol=1e-6, what="kanal başına ölçekli gürültü (kanallar arası aynı harita)")
        wt = torch.randn(5, 6, 3, 3)
        s = torch.rand(4, 6) + 0.5
        out = need(ns, "modulated_conv")(x, wt, s, False)
        ref = torch.stack([F.conv2d((x[i:i + 1] * s[i][None, :, None, None]), wt, padding=1)[0] for i in range(4)])
        close(out.numpy(), ref.numpy(), atol=1e-4, what="modülasyon = girdiyi kanal başına ölçeklemek")
        xs = torch.randn(4, 6, 16, 16)
        od = ns["modulated_conv"](xs, wt, s * 10, True)
        close(od.std((0, 2, 3)).numpy(), torch.ones(5).numpy(), atol=0.15, what="demodülasyon: i.i.d. birim girdide çıktı std ≈ 1, stil ölçeğinden bağımsız")
        W = M(torch.randn(2000, 16)).detach()
        wavg = W.mean(0)
        rows = []
        for psi in (1.0, 0.7, 0.5, 0.0):
            Wt = need(ns, "truncate")(W, wavg, psi)
            rows.append([psi, f"{Wt.std(0).mean().item():.4f}", f"{(Wt - wavg).norm(dim=1).mean().item():.4f}"])
        ensure(float(rows[-1][1]) < 1e-6, "ψ = 0: tek görüntü (ortalama)")
        R_.table("Kırpma hilesi: w uzayında çeşitlilik", ["ψ", "boyut başına std", "ort. ‖w − w̄‖"], rows,
                 note="Kırpma w'leri ortalamaya çeker: düşük olasılıklı (kötü üretilen) bölgelerden kaçınır, kalite artar, çeşitlilik ψ ile doğrusal azalır. StyleGAN2'de AdaIN'in istatistik normalizasyonu ağırlıklara taşınır (modülasyon + demodülasyon): aynı stil kontrolü, 'damla' (droplet) artefaktları olmadan.")
    R_.test("StyleGAN blokları", t)
