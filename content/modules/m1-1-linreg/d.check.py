import torch
from torch import nn
from labcheck import close, need, ensure, uses_any


def check(ns, R):
    src = ns["__source__"]

    def g():
        out = need(ns, "one_step_grads")()
        ensure(isinstance(out, (tuple, list)) and len(out) == 2, "(gw, gb) döndürmeli")
        close(out, [-50 / 3, -22 / 3], rtol=1e-5)
    R.test("one_step_grads: (−16.67, −7.33) — elle bulduğunla aynı", g)
    R.test("autograd + optim kullanılıyor", lambda: ensure(uses_any(src, ["backward"]) and uses_any(src, ["SGD", "step"]), "loss.backward() ve opt.step() bekleniyordu"))

    def fit():
        torch.manual_seed(0)
        X = torch.randn(120, 3)
        y = X @ torch.tensor([2.0, -1.0, 0.5]) + 3 + 0.05 * torch.randn(120)
        Xt = torch.cat([X, torch.ones(120, 1)], 1).double()
        th = torch.linalg.lstsq(Xt, y.double().unsqueeze(1)).solution.squeeze(1)
        m = need(ns, "fit_torch")(X, y, lr=0.1, steps=500)
        ensure(isinstance(m, nn.Linear), "nn.Linear döndürmeli")
        close(m.weight.detach().reshape(-1), th[:3], atol=2e-3, rtol=0, what="weight")
        close(m.bias.detach().reshape(-1), th[3:], atol=2e-3, rtol=0, what="bias")
        return "ağırlıklar normal denklem çözümüyle 3 basamak eşleşti"
    R.test("fit_torch normal denklem çözümüne ulaşıyor", fit)
