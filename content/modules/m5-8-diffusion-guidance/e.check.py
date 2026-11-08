import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from labcheck import need, ensure


def check(ns, R_):
    def t():
        X, y = load_digits(return_X_y=True)
        X = torch.tensor(X, dtype=torch.float32) / 16
        y = torch.tensor(y)
        torch.manual_seed(0)
        enc = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 8))
        dec = nn.Sequential(nn.Linear(8, 128), nn.ReLU(), nn.Linear(128, 64), nn.Sigmoid())
        o = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), 3e-3)
        for _ in range(150):
            p = torch.randperm(len(X))
            for i in range(0, len(X), 64):
                b = X[p[i:i + 64]]
                loss = F.mse_loss(dec(enc(b)), b)
                o.zero_grad()
                loss.backward()
                o.step()
        clf = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10))
        o = torch.optim.Adam(clf.parameters(), 1e-3)
        for _ in range(100):
            p = torch.randperm(len(X))
            for i in range(0, len(X), 64):
                b = p[i:i + 64]
                loss = F.cross_entropy(clf(X[b]), y[b])
                o.zero_grad()
                loss.backward()
                o.step()
        with torch.no_grad():
            Z = enc(X)
            mu, sd = Z.mean(0), Z.std(0)
            Z = (Z - mu) / sd
        labels = torch.arange(10).repeat(50)
        rows, R = [], {}
        for name, data, D in (("piksel uzayı (64)", X * 2 - 1, 64), ("gizli uzay (8) + kod çözücü", Z, 8)):
            t0 = time.perf_counter()
            m = need(ns, "train_diffusion")(data, y, 3000, 0)
            tt = time.perf_counter() - t0
            for s in (1.0, 3.0):
                t0 = time.perf_counter()
                z = need(ns, "sample")(m, labels, D, s, 1)
                ts = time.perf_counter() - t0
                with torch.no_grad():
                    img = ((z + 1) / 2).clamp(0, 1) if D == 64 else dec(z * sd + mu)
                    pr = clf(img).softmax(1)
                acc = (pr.argmax(1) == labels).float().mean().item()
                R[(D, s)] = acc
                rows.append([name, s, f"{tt:.1f} sn", f"{ts:.2f} sn", f"{acc:.3f}", f"{pr.max(1).values.mean().item():.3f}"])
        ensure(R[(8, 3.0)] > 0.85 and R[(8, 3.0)] > R[(64, 3.0)] + 0.2, f"gizli uzay aynı bütçede çok daha iyi {R}")
        ensure(R[(8, 3.0)] >= R[(8, 1.0)] - 0.02, "CFG sınıf doğruluğunu korur/artırır")
        R_.table("Koşullu difüzyon, rakamlar, 3000 adım (aynı MLP bütçesi), 500 örnek", ["uzay", "s", "eğitim", "örnekleme", "sınıf doğruluğu", "ort. güven"], rows,
                 note="Gizli uzay 8 kat küçük ve otomatik kodlayıcı piksel düzeyindeki ayrıntıyı (kalem kalınlığı, gürültü) zaten üstlenmiş: difüzyon modeli yalnız 'hangi rakam, hangi stil' yapısını öğrenir. Aynı eğitim bütçesinde piksel uzayındaki model çok geride kalır. Gerçek latent diffusion'da U-Net ve 8× küçültme ile hesap kazancı da büyüktür (burada MLP gizli boyutu baskın olduğundan süreler yakın).")
    R_.test("piksel vs gizli uzay", t)
