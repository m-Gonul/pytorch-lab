import time
import torch
import torch.nn.functional as F
from sklearn.datasets import load_digits
from labcheck import need, ensure


def check(ns, R_):
    def t():
        X = torch.tensor(load_digits().data, dtype=torch.float32).view(-1, 1, 8, 8) / 16
        X = F.interpolate(X, size=16, mode="bilinear", align_corners=False)
        g = torch.Generator().manual_seed(0)
        p = torch.randperm(len(X), generator=g)
        tr, te = X[p[497:]], X[p[:497]]
        base = F.mse_loss(tr.mean(0, keepdim=True).expand_as(te), te).item()
        rows = []
        res = {}
        for name in ("ConvAE", "DenseAE"):
            torch.manual_seed(0)
            m = need(ns, name)(16)
            ensure(tuple(m(te[:2]).shape) == (2, 1, 16, 16), f"{name} çıkış şekli")
            opt = torch.optim.Adam(m.parameters(), 3e-3)
            t0 = time.perf_counter()
            for _ in range(60):
                perm = torch.randperm(len(tr))
                for i in range(0, len(tr), 64):
                    xb = tr[perm[i:i + 64]]
                    loss = F.mse_loss(m(xb), xb)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            with torch.no_grad():
                mse = F.mse_loss(m(te), te).item()
            res[name] = (sum(q.numel() for q in m.parameters()), mse)
            rows.append([name, res[name][0], f"{mse:.4f}", f"{time.perf_counter() - t0:.1f} sn"])
        ensure(res["ConvAE"][1] < base / 3 and res["DenseAE"][1] < base / 3, "ikisi de ortalama görüntüden çok iyi")
        ensure(res["ConvAE"][0] < res["DenseAE"][0], "konvolüsyonel daha az parametre")
        R_.table(f"16 × 16 rakamlar, 16 boyutlu kod, 60 epoch (ortalama görüntü MSE = {base:.4f})", ["model", "parametre", "test MSE", "süre"], rows,
                 note="Konvolüsyonel kodlayıcı uzamsal boyutu azaltırken derinliği artırır, kod çözücü transposed conv ile tersini yapar. Parametrelerin çoğu her iki modelde de kod ile düzleştirilmiş harita arasındaki yoğun katmandadır. Büyük görüntülerde yoğun AE'nin parametresi piksel sayısıyla patlar; konvolüsyonel olan ölçeklenir.")
    R_.test("konvolüsyonel vs yoğun", t)
