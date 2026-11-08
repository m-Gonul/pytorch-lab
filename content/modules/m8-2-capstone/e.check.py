import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from labcheck import close, need, ensure


class _Net(nn.Module):
    def __init__(self, c=16, C=10):
        super().__init__()
        self.C = C
        self.emb = nn.Embedding(C + 1, c)
        self.body = nn.Sequential(nn.Conv2d(1 + c, c, 3, padding=1), nn.SiLU(), nn.Conv2d(c, c, 3, padding=1), nn.SiLU(), nn.Conv2d(c, 1, 3, padding=1))

    def forward(self, x, t, y):
        e = self.emb(y)[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3]) + (t.float() / 200)[:, None, None, None]
        return self.body(torch.cat([x, e], 1))


def check(ns, R_):
    def t():
        torch.manual_seed(0)
        net = _Net().eval()
        x = torch.randn(8, 1, 16, 16)
        tt = torch.full((8,), 37)
        y = torch.arange(8)
        with torch.no_grad():
            a = need(ns, "cfg_two_calls")(net, x, tt, y, 2.5)
            b = need(ns, "cfg_batched")(net, x, tt, y, 2.5)
        close(a.numpy(), b.numpy(), atol=1e-5, what="toplu CFG = iki çağrı")
        labels = torch.arange(10).repeat(20)
        d1, d2 = need(ns, "ddim_sample")(net, labels, 2.0, 25, 1), ns["ddim_sample"](net, labels, 2.0, 25, 1)
        ensure(torch.equal(d1, d2), "DDIM deterministik (aynı tohum)")
        p1, p2 = need(ns, "ddpm_sample")(net, labels, 2.0, 1), ns["ddpm_sample"](net, labels, 2.0, 2)
        ensure(tuple(p1.shape) == (200, 1, 16, 16) and not torch.equal(p1, p2), "DDPM stokastik")
        rows = []
        with torch.no_grad():
            for name, f in (("iki çağrı", ns["cfg_two_calls"]), ("toplu", ns["cfg_batched"])):
                f(net, x, tt, y, 2.0)
                t0 = time.perf_counter()
                for _ in range(20):
                    f(net, torch.randn(64, 1, 16, 16), torch.full((64,), 10), torch.arange(64) % 10, 2.0)
                rows.append([f"CFG {name} (batch 64, 20 kez)", f"{time.perf_counter() - t0:.3f} sn"])
        for name, f in (("DDPM 200 adım", lambda: ns["ddpm_sample"](net, labels, 2.0, 0)), ("DDIM 25 adım", lambda: ns["ddim_sample"](net, labels, 2.0, 25, 0)), ("DDIM 10 adım", lambda: ns["ddim_sample"](net, labels, 2.0, 10, 0))):
            t0 = time.perf_counter()
            f()
            rows.append([f"{name} (200 örnek)", f"{time.perf_counter() - t0:.3f} sn"])
        R_.table("Çıkarım maliyeti (CPU, küçük ağ)", ["işlem", "süre"], rows,
                 note="Koşullu ve koşulsuz tahmini tek batch'te hesaplamak çağrı ve çekirdek başlatma yükünü yarıya indirir (GPU'da büyük batch verimli olduğu için kazanç daha da büyüktür). Örnekleme süresi adım sayısıyla doğrusal: d aşamasında DDIM-10'un kalitesi DDIM-25'e yakındı, DDPM-200'ün 20 katı daha ucuz. Damıtma (progressive distillation, consistency models) bu sayıyı 1–4 adıma indirir.")
    R_.test("çıkarım mühendisliği", t)
