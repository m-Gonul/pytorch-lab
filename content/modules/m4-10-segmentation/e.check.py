import time
import torch
from labcheck import need, ensure, fmt_time


def check(ns, R_):
    def t():
        torch.manual_seed(0)
        U = need(ns, "upsamplers")(32)
        ensure(set(U) == {"tconv k3 s2", "tconv k4 s2", "nearest + conv3", "bilinear + conv3", "PixelShuffle"}, "beş yöntem")
        x = torch.randn(8, 32, 64, 64)
        rows, art = [], {}
        for name, m in U.items():
            ensure(tuple(m(x[:1]).shape) == (1, 32, 128, 128), f"{name}: ×2")
            with torch.no_grad():
                m(x)
                best = 1e9
                for _ in range(3):
                    t0 = time.perf_counter()
                    m(x)
                    best = min(best, time.perf_counter() - t0)
            nparam = sum(p.numel() for p in m.parameters())
            art[name] = need(ns, "structural_artifact")(m)
            rows.append([name, nparam, fmt_time(best), f"{art[name]:.3f}"])
        ensure(art["tconv k3 s2"] > 0.5 and art["tconv k4 s2"] < 1e-5 and art["nearest + conv3"] < 1e-5, "yapısal artefakt: yalnız k3 s2")
        R_.table("×2 yukarı örnekleme, 32 kanal, 8 × 64² → 128² (CPU)", ["yöntem", "parametre", "süre", "yapısal artefakt"], rows,
                 note="'Yapısal artefakt' = eşit ağırlıklar ve sabit girdiyle iç bölgede (en büyük − en küçük)/ortalama. k = 3, s = 2'lik tconv'da katkı sayısı 1-2-4 diye değiştiği için sabit girdi bile dama tahtası olur; k = 4 ve 'yeniden boyutlandır + konv.' düzgündür. PixelShuffle eşit ağırlıkla düzgündür ama rastgele başlatmada dört faz bağımsızdır (ICNR başlatması bunu giderir).")
    R_.test("yukarı örnekleyiciler", t)
