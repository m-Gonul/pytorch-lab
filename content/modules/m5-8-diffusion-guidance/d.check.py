import time
import torch
from labcheck import need, ensure


def check(ns, R_):
    def t():
        MU, LAB = ns["MU"], ns["LAB"]
        torch.manual_seed(0)
        m = need(ns, "EpsNet")(4, 128)
        ensure(tuple(m(torch.randn(3, 2), torch.tensor([0, 5, 199]), torch.tensor([0, 4, 2])).shape) == (3, 2), "çıkış (B, 2)")
        t0 = time.perf_counter()
        need(ns, "train")(m, 4000, 0.15, 0)
        tt = time.perf_counter() - t0
        rows, R = [], {}
        for s in (0.0, 1.0, 3.0, 6.0):
            x = need(ns, "sample")(m, 2000, 1, s, 1)
            d2 = torch.cdist(x, MU) ** 2
            r = torch.softmax(-d2 / 0.5, 1)
            py = r[:, LAB == 1].sum(1).mean().item()
            R[s] = (py, x.norm(dim=1).mean().item(), d2.min(1).values.sqrt().median().item())
            rows.append([s, f"{py:.3f}", f"{R[s][1]:.2f}", f"{R[s][2]:.2f}"])
        ensure(R[0.0][0] < 0.5 and R[1.0][0] > 0.9, f"koşulsuz/koşullu {R}")
        ensure(R[6.0][1] > R[1.0][1] + 0.5 and R[6.0][2] > R[1.0][2], "büyük s: dışa taşma")
        R_.table(f"Etiket bırakmalı (%15) koşullu ağ, 4000 adım ({tt:.0f} sn); hedef sınıf 1", ["s", "ort. p(y = 1 | x)", "ort. yarıçap (veri 4)", "medyan en yakın merkez uzaklığı"], rows,
                 note="Tek ağ, boş token'la koşulsuz, etiketle koşullu ε verir. s = 0 sınıfı yok sayar (≈ 0.25), s = 1 koşullu dağılımdan örnekler; s arttıkça c aşamasındaki kesin modeldeki gibi örnekler sınıf ayrımı yönünde dışa itilir. Öğrenilmiş modelde koşullu ε'nun kusurları yönlendirmeyle büyütülür — pratikte s ≈ 3–8 (görüntü) bu yüzden ayarlanır.")
    R_.test("CFG ile koşullu difüzyon", t)
