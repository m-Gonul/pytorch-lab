import torch
import torch.nn.functional as F
from torch import nn
from labcheck import close, need, ensure


def check(ns, R):
    def drop():
        D = need(ns, "MyDropout", "sınıf")(0.3)
        x = torch.ones(200_000, requires_grad=True)
        torch.manual_seed(0)
        y = D(x)
        frac = (y == 0).float().mean().item()
        ensure(abs(frac - 0.3) < 0.01, f"sıfır oranı {frac:.3f}, 0.3 bekleniyordu")
        kept = y[y != 0]
        close(kept[:5], torch.full((5,), 1 / 0.7), what="ölçek 1/(1−p)")
        y.sum().backward()
        ensure(bool(((x.grad == 0) == (y.detach() == 0)).all()), "düşen elemanların gradyanı 0 olmalı")
        D.eval()
        z = torch.randn(10)
        ensure(torch.equal(D(z), z), "eval'de kimlik")
        return f"sıfır oranı {frac:.4f}"
    R.test("MyDropout: oran, ölçek, gradyan, eval", drop)

    def es():
        g = torch.Generator().manual_seed(0)
        n = 60
        Xtr = torch.randn(n, 10, generator=g); ytr = (Xtr[:, 0] + 0.8 * torch.randn(n, generator=g) > 0).long()
        Xva = torch.randn(400, 10, generator=g); yva = (Xva[:, 0] + 0.8 * torch.randn(400, generator=g) > 0).long()
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(10, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 2))
        m, best_epoch, hist = need(ns, "fit_early_stopping")(model, Xtr, ytr, Xva, yva, patience=10, max_epochs=500, lr=1e-2)
        ensure(len(hist) < 500, "aşırı uyuma giden modelde erken durmalıydı")
        ensure(best_epoch == int(torch.tensor(hist).argmin()), "best_epoch doğrulama kaybının argmin'i olmalı")
        ensure(len(hist) - 1 - best_epoch == 10, f"patience = 10: en iyiden sonra {len(hist) - 1 - best_epoch} epoch")
        m.eval()
        with torch.no_grad():
            v = F.cross_entropy(m(Xva), yva).item()
        close(v, min(hist), rtol=1e-6, what="döndürülen modelin doğrulama kaybı = en iyi")
        return f"{len(hist)} epoch'ta durdu; en iyi epoch {best_epoch}, val kaybı {min(hist):.3f} (son {hist[-1]:.3f})"
    R.test("fit_early_stopping: durur ve en iyi ağırlıkları geri yükler", es)
