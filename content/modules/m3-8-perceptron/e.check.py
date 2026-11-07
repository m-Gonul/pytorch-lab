import torch
from labcheck import need, ensure, bench, fmt_time


def check(ns, R_):
    def t():
        m = need(ns, "BatchedMLP")(5, 3, 8, 2, 2)
        ensure(tuple(m(torch.randn(5, 7, 3)).shape) == (5, 7, 2), "şekil")
        x = torch.randn(1, 4, 3).expand(5, 4, 3)
        out = m(x)
        ensure(not torch.allclose(out[0], out[1]), "her model kendi ağırlıkları")
        rows = []
        for fn in ("f1", "f2"):
            r = {d: need(ns, "train_batched")(16, fn, d, 1200) for d in (1, 2, 3)}
            for d, v in r.items():
                q = torch.quantile(v, torch.tensor([0.1, 0.5, 0.9]))
                rows.append([fn, d, f"{float(q[1]):.2e}", f"[{float(q[0]):.1e}, {float(q[2]):.1e}]"])
            ensure(float(r[3].median()) < float(r[1].median()), f"{fn}: ortanca hata derinlikle düşmeli")
        tb = bench(lambda: ns["train_batched"](16, "f2", 2, 300), repeat=1, number=1)
        tl = bench(lambda: need(ns, "train_loop")(4, "f2", 2, 300), repeat=1, number=1) * 4
        R_.table("16 model × (veri tohumu), genişlik 64", ["fonksiyon", "derinlik", "ortanca test MSE", "%10–%90 aralığı"], rows,
                 note=f"Süre (300 adım, 16 model): toplu {fmt_time(tb)}, döngü (tahmini) {fmt_time(tl)}. Tek bir tohumla yapılan karşılaştırma (d aşaması) yanıltıcı olabilir; dağılım raporla.")
    R_.test("toplu eğitim", t)
