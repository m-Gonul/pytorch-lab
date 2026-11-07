import ast
import torch
import torch.nn.functional as F
from torch import nn
from labcheck import close, need, ensure


def check(ns, R):
    src = ns["__source__"]

    def nb():
        fn = next((n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "svm_loss_torch"), None)
        ensure(fn is not None, "`svm_loss_torch` yok")
        ensure("multi_margin_loss" not in ast.unparse(fn), "svm_loss_torch'u elle yaz")
    R.test("hazır multi_margin_loss kullanılmıyor", nb)
    g = torch.Generator().manual_seed(0)
    S = torch.randn(25, 6, generator=g, dtype=torch.float64) * 2
    y = torch.randint(0, 6, (25,), generator=g)
    R.test("= C · F.multi_margin_loss", lambda: close(need(ns, "svm_loss_torch")(S, y), 6 * F.multi_margin_loss(S, y)))
    R.test("elle örnek 2.9", lambda: close(need(ns, "svm_loss_torch")(torch.tensor([[3.2, 5.1, -1.7]]), torch.tensor([0])), 2.9))

    def tr():
        centers = torch.tensor([[0.0, 3.0], [-3.0, -2.0], [3.0, -2.0]])
        yy = torch.arange(3).repeat_interleave(60)
        X = centers[yy] + torch.randn(180, 2, generator=g)
        torch.manual_seed(0)
        m = need(ns, "train_linear_svm")(X, yy, 3)
        ensure(isinstance(m, nn.Linear), "nn.Linear bekleniyordu")
        acc = (m(X).argmax(1) == yy).float().mean().item()
        ensure(acc >= 0.9, f"doğruluk {acc:.2f}")
        return f"doğruluk {acc:.3f}"
    R.test("train_linear_svm", tr)
