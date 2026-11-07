import ast
import torch
import torch.nn.functional as F
from torch import nn
from labcheck import close, need, ensure


def check(ns, R):
    src = ns["__source__"]

    def no_builtin():
        t = ast.parse(src)
        for node in ast.walk(t):
            if isinstance(node, ast.FunctionDef) and node.name == "bce_logits":
                names = {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)} | {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
                ensure(not ({"binary_cross_entropy_with_logits", "binary_cross_entropy", "BCEWithLogitsLoss", "BCELoss"} & names), "bce_logits'i hazır kayıp fonksiyonuyla yazma; formülü kendin uygula")
                return
        raise AssertionError("`bce_logits` bulunamadı")
    R.test("bce_logits hazır kayıp fonksiyonu kullanmıyor", no_builtin)

    z = torch.tensor([-1.0, 0.0, 2.0, -100.0, 100.0, 30.0], dtype=torch.float64)
    y = torch.tensor([0.0, 1.0, 1.0, 1.0, 0.0, 1.0], dtype=torch.float64)

    def val():
        out = need(ns, "bce_logits")(z, y)
        ensure(torch.isfinite(out), f"sonuç sonlu değil: {out}")
        close(out, F.binary_cross_entropy_with_logits(z, y))
        close(need(ns, "bce_logits")(z[:3], y[:3]), 0.3777789, atol=1e-6, what="elle örnek")
    R.test("bce_logits PyTorch'unkiyle eşleşiyor (z = ±100 dahil)", val)

    def gr():
        zz = z.clone().requires_grad_(True)
        need(ns, "bce_logits")(zz, y).backward()
        close(zz.grad, (torch.sigmoid(z) - y) / len(z), what="∂L/∂z")
        return "∂L/∂z = (σ(z) − y)/n, z = 0 dahil"
    R.test("autograd gradyanı (σ(z) − y)/n — z = 0 noktasında da", gr)

    def tr():
        g = torch.Generator().manual_seed(1)
        X = torch.cat([torch.randn(100, 2, generator=g) - 1.3, torch.randn(100, 2, generator=g) + 1.3])
        t = torch.cat([torch.zeros(100), torch.ones(100)])
        torch.manual_seed(0)
        m = need(ns, "train_torch")(X, t, lr=0.5, steps=400)
        ensure(isinstance(m, nn.Linear), "nn.Linear döndürmeli")
        with torch.no_grad():
            acc = ((m(X).squeeze(1) > 0).float() == t).float().mean().item()
        ensure(acc >= 0.9, f"doğruluk {acc:.2f}")
        return f"doğruluk {acc:.3f}"
    R.test("train_torch iki kümeyi ayırıyor", tr)
