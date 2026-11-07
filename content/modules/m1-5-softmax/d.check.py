import ast
import torch
import torch.nn.functional as F
from torch import nn
from labcheck import close, need, ensure


def check(ns, R):
    src = ns["__source__"]

    def nobuiltin():
        fn = next((n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "ce_from_logits"), None)
        ensure(fn is not None, "`ce_from_logits` bulunamadı")
        names = {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)} | {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
        ensure(not ({"cross_entropy", "log_softmax", "CrossEntropyLoss", "nll_loss", "softmax"} & names), "ce_from_logits'i hazır fonksiyonlarla yazma")
    R.test("ce_from_logits hazır CE/softmax kullanmıyor", nobuiltin)
    g = torch.Generator().manual_seed(0)
    Z = torch.randn(16, 5, generator=g, dtype=torch.float64) * 3
    y = torch.randint(0, 5, (16,), generator=g)

    def val():
        f = need(ns, "ce_from_logits")
        close(f(Z, y), F.cross_entropy(Z, y))
        Zb = Z.clone(); Zb[0] += 1e4; Zb[1, 2] = -1e4
        v = f(Zb, y)
        ensure(torch.isfinite(v), "büyük logit'lerde sonuç sonlu değil")
        close(v, F.cross_entropy(Zb, y), what="±10⁴ logit")
    R.test("F.cross_entropy ile aynı, ±10⁴'te sonlu", val)

    def grad():
        Zg = Z.clone().requires_grad_(True)
        need(ns, "ce_from_logits")(Zg, y).backward()
        close(Zg.grad, (torch.softmax(Z, 1) - F.one_hot(y, 5)) / 16, what="∂L/∂Z")
    R.test("autograd gradyanı (P − Y)/n", grad)

    def tr():
        centers = torch.tensor([[0.0, 3.0], [-3.0, -2.0], [3.0, -2.0], [0.0, -5.0]])
        yy = torch.arange(4).repeat_interleave(50)
        X = centers[yy] + torch.randn(200, 2, generator=g)
        torch.manual_seed(0)
        m = need(ns, "train_softmax")(X, yy, 4)
        ensure(isinstance(m, nn.Linear) and m.out_features == 4, "nn.Linear(d, 4) bekleniyordu")
        acc = (m(X).argmax(1) == yy).float().mean().item()
        ensure(acc >= 0.85, f"doğruluk {acc:.2f}")
        return f"4 sınıf, doğruluk {acc:.3f}"
    R.test("train_softmax", tr)
