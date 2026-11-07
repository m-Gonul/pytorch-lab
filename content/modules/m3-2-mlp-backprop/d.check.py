import math
import torch
from torch import nn
from labcheck import close, need, ensure


def spiral(n=100, K=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    X, y = [], []
    for k in range(K):
        r = torch.linspace(0.0, 1, n)
        t = torch.linspace(k * 4, (k + 1) * 4, n) + torch.randn(n, generator=g) * 0.2
        X.append(torch.stack([r * t.sin(), r * t.cos()], 1)); y += [k] * n
    return torch.cat(X), torch.tensor(y)


def check(ns, R):
    g = torch.Generator().manual_seed(0)

    def gr():
        X = torch.randn(9, 3, generator=g, dtype=torch.float64); y = torch.randint(0, 4, (9,), generator=g)
        W1, b1 = torch.randn(3, 6, generator=g, dtype=torch.float64), torch.randn(6, generator=g, dtype=torch.float64)
        W2, b2 = torch.randn(6, 4, generator=g, dtype=torch.float64), torch.randn(4, generator=g, dtype=torch.float64)
        ps = [t.clone().requires_grad_(True) for t in (W1, b1, W2, b2)]
        L = need(ns, "two_layer_torch")(X, y, *ps, 0.05)
        L.backward()
        Z = X @ W1 + b1; H = Z.clamp(min=0); S = H @ W2 + b2
        dS = (torch.softmax(S, 1) - torch.nn.functional.one_hot(y, 4)) / 9
        dW2 = H.T @ dS + 0.1 * W2; db2 = dS.sum(0)
        dZ = (dS @ W2.T) * (Z > 0)
        dW1 = X.T @ dZ + 0.1 * W1; db1 = dZ.sum(0)
        for p, want, n in zip(ps, (dW1, db1, dW2, db2), ("W1", "b1", "W2", "b2")):
            close(p.grad, want, what=f"d{n}")
        return "autograd = elle türetilen matris formülleri"
    R.test("autograd gradyanları matris formülleriyle aynı", gr)

    def tr():
        X, y = spiral()
        torch.manual_seed(0)
        m = need(ns, "train_mlp")(X, y, 100, 3)
        ensure(isinstance(m, nn.Module), "nn.Module bekleniyordu")
        acc = (m(X).argmax(1) == y).float().mean().item()
        ensure(acc >= 0.95, f"spiral doğruluğu {acc:.2f}")
        return f"spiral: doğruluk {acc:.3f}"
    R.test("train_mlp spiral veride ≥ 0.95", tr)
