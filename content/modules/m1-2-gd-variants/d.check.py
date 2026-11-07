import torch
from torch import nn
from labcheck import close, need, ensure, uses_any


def check(ns, R):
    src = ns["__source__"]
    R.test("DataLoader ve optim.SGD kullanılıyor", lambda: ensure(uses_any(src, ["DataLoader"]) and uses_any(src, ["SGD"]), "DataLoader + torch.optim.SGD bekleniyordu"))

    def path():
        p = need(ns, "momentum_path_torch")(0.0, 0.25, 0.5, 5)
        w, v, ref = 0.0, 0.0, []
        for _ in range(5):
            v = 0.5 * v + 2 * (w - 3); w -= 0.25 * v; ref.append(w)
        close(p, ref, rtol=1e-6)
        return "PyTorch: " + ", ".join(f"{t:.4f}" for t in p)
    R.test("SGD(momentum) yörüngesi elle formülle aynı", path)

    def tr():
        torch.manual_seed(0)
        X = torch.randn(600, 4)
        y = X @ torch.tensor([1.0, -2.0, 0.5, 0.0]) + 1.0 + 0.05 * torch.randn(600)
        m = need(ns, "train_torch")(X, y, lr=0.02, batch_size=32, epochs=40, momentum=0.9)
        ensure(isinstance(m, nn.Linear), "nn.Linear döndürmeli")
        Xt = torch.cat([X, torch.ones(600, 1)], 1)
        th = torch.linalg.lstsq(Xt, y.unsqueeze(1)).solution.squeeze()
        close(m.weight.detach().view(-1), th[:4], atol=0.03, rtol=0, what="weight")
        close(m.bias.detach(), th[4:], atol=0.03, rtol=0, what="bias")
    R.test("train_torch en küçük kareler çözümüne ulaşıyor", tr)
