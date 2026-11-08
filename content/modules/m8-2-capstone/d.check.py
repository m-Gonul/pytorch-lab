import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from labcheck import need, ensure


def _fid(a, b):
    m1, m2 = a.mean(0), b.mean(0)
    S1, S2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    w, V = np.linalg.eigh(S1)
    r = (V * np.sqrt(np.clip(w, 0, None))) @ V.T
    return float(((m1 - m2) ** 2).sum() + np.trace(S1 + S2) - 2 * np.sqrt(np.clip(np.linalg.eigvalsh(r @ S2 @ r), 0, None)).sum())


def check(ns, R_):
    def t_arch():
        torch.manual_seed(0)
        m = need(ns, "UNet")(16, 10)
        x = torch.randn(3, 1, 16, 16)
        out = m(x, torch.tensor([0, 50, 199]), torch.tensor([1, 2, 10]))
        ensure(tuple(out.shape) == (3, 1, 16, 16), "çıkış (B, 1, 16, 16)")
        rb = need(ns, "ResBlock")(8, 8, 32)
        z = torch.randn(2, 8, 4, 4)
        ensure(torch.allclose(rb(z, torch.randn(2, 32)), z), "sıfır başlatılmış blok özdeşlik")
        at = need(ns, "Attn")(8)
        ensure(torch.allclose(at(z), z), "attention başlangıçta özdeşlik")
        ensure(m.cemb.num_embeddings == 11, "10 sınıf + boş token")
    R_.test("mimari", t_arch)

    def t_train():
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
        feat = lambda z: clf[:-1](z).detach().numpy()
        fr = feat(X)
        base = _fid(feat(X[::2]), feat(X[1::2]))
        torch.manual_seed(0)
        t0 = time.perf_counter()
        m, ema = need(ns, "train")(ns["UNet"](16, 10), 1200, 0, 0.99)
        tt = time.perf_counter() - t0
        labels = torch.arange(10).repeat(50)
        rows, R = [], {}
        for name, net in (("ham ağırlıklar", m), ("EMA (0.99)", ema)):
            for s, steps in ((1.0, 25), (2.0, 25), (1.0, 10)):
                t1 = time.perf_counter()
                xs = need(ns, "sample")(net, labels, s, steps, 1)
                ts = time.perf_counter() - t1
                with torch.no_grad():
                    pr = clf(xs).softmax(1)
                acc = (pr.argmax(1) == labels).float().mean().item()
                R[(name, s, steps)] = (acc, _fid(fr, feat(xs)))
                rows.append([name, s, steps, f"{ts:.1f} sn", f"{acc:.3f}", f"{R[(name, s, steps)][1]:.1f}"])
        e1, r1 = R[("EMA (0.99)", 1.0, 25)], R[("ham ağırlıklar", 1.0, 25)]
        ensure(e1[1] < r1[1] and e1[0] > 0.75, f"EMA daha iyi FID ve makul sınıf doğruluğu {R}")
        ensure(R[("EMA (0.99)", 2.0, 25)][0] > e1[0], "yönlendirme sınıf doğruluğunu artırır")
        R_.table(f"Koşullu U-Net difüzyonu, 16 × 16 rakamlar, 1200 adım ({tt:.0f} sn) — FID taban çizgisi (gerçek–gerçek) {base:.1f}", ["ağırlıklar", "s", "DDIM adımı", "örnekleme (500)", "sınıf doğruluğu", "FID"], rows,
                 note="Aynı eğitim koşusunda EMA ağırlıkları FID'i belirgin düşürür: difüzyon örneklemesi ağı onlarca kez zincirlediği için ağırlıklardaki SGD gürültüsü örneklerde birikir. s = 2 sınıf doğruluğunu artırır ama FID'i iyileştirmeyebilir (çeşitlilik ↓, aşırı doyma). DDIM'de 10 adım bu küçük problemde 25'e yakın kalite verir.")
    R_.test("eğitim ve değerlendirme", t_train)
