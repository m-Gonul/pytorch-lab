import ast
import torch
from labcheck import close, need, ensure


def ridge_cf(X, y, lam2):
    n = X.shape[0]
    Xc, yc = X - X.mean(0), y - y.mean()
    w = torch.linalg.solve(Xc.T @ Xc / n + lam2 * torch.eye(X.shape[1], dtype=X.dtype), Xc.T @ yc / n)
    return w, y.mean() - X.mean(0) @ w


def check(ns, R):
    src = ns["__source__"]
    torch.manual_seed(0)
    X = torch.randn(120, 3)
    y = X @ torch.tensor([1.0, -2.0, 0.3]) + 0.5 + 0.05 * torch.randn(120)
    w_cf, b_cf = ridge_cf(X.double(), y.double(), 0.2)

    def fr():
        torch.manual_seed(1)
        m = need(ns, "fit_reg")(X, y, 0.2)
        close(m.weight.detach().view(-1), w_cf, atol=2e-3, rtol=0, what="weight")
        close(m.bias.detach(), b_cf.view(1), atol=2e-3, rtol=0, what="bias (cezasız)")
    R.test("fit_reg (λ₂ = 0.2) ridge kapalı formuyla aynı", fr)

    def l1():
        torch.manual_seed(1)
        m = need(ns, "fit_reg")(X, y, 0.0, lam1=0.5, steps=3000, lr=0.05)
        w = m.weight.detach().view(-1)
        ensure(abs(float(w[2])) < 0.05, f"küçük gerçek katsayı (0.3) L1 ile bastırılmalı: {float(w[2]):.3f}")
        return "L1 ağırlıklar: " + ", ".join(f"{float(v):.3f}" for v in w) + " (SGD tam 0 vermez; yalnız küçültür)"
    R.test("fit_reg L1: küçük katsayıyı bastırıyor", l1)

    def wd():
        tree = ast.parse(src)
        fn = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "fit_weight_decay"), None)
        ensure(fn is not None, "`fit_weight_decay` bulunamadı")
        names = {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
        ensure("abs" not in names, "fit_weight_decay cezayı kayba eklememeli")
        txt = ast.unparse(fn)
        ensure("weight_decay" in txt, "weight_decay kullanılmıyor")
        torch.manual_seed(1)
        m = need(ns, "fit_weight_decay")(X, y, 0.2)
        close(m.weight.detach().view(-1), w_cf, atol=2e-3, rtol=0, what="weight")
        close(m.bias.detach(), b_cf.view(1), atol=2e-3, rtol=0, what="bias (decay'siz olmalı)")
        return "weight_decay = 2λ ve bias muaf"
    R.test("fit_weight_decay aynı ridge çözümüne ulaşıyor", wd)
