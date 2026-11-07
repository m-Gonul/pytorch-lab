import torch
from torch.autograd import gradcheck
from labcheck import close, need, ensure


def check(ns, R):
    LSE = need(ns, "LogSumExp", "sınıf")

    def fw():
        x = torch.randn(3, 4, 5, dtype=torch.float64)
        close(LSE.apply(x), torch.logsumexp(x, -1))
        big = torch.tensor([[1000.0, 1000.0, 0.0]])
        ensure(torch.isfinite(LSE.apply(big)).all(), "büyük girdide sonlu olmalı")
    R.test("LogSumExp ileri = torch.logsumexp", fw)

    def gc():
        x = torch.randn(3, 6, dtype=torch.float64, requires_grad=True)
        ensure(gradcheck(LSE.apply, (x,)), "gradcheck")
        b = torch.tensor([[1000.0, 999.0]], dtype=torch.float64, requires_grad=True)
        LSE.apply(b).sum().backward()
        close(b.grad, torch.softmax(b.detach(), -1), what="büyük girdide gradyan")
    R.test("LogSumExp gradcheck + kararlılık", gc)

    def ste():
        x = torch.tensor([0.3, 1.7, -0.6], requires_grad=True)
        y = need(ns, "RoundSTE", "sınıf").apply(x)
        close(y, [0.0, 2.0, -1.0])
        (y * torch.tensor([2.0, 3.0, 4.0])).sum().backward()
        close(x.grad, [2.0, 3.0, 4.0], what="STE gradyanı")
    R.test("RoundSTE", ste)

    def gr():
        x = torch.tensor([1.0, -2.0], requires_grad=True)
        y = need(ns, "GradReverse", "sınıf").apply(x, 0.5)
        close(y, x.detach(), what="ileri özdeşlik")
        (y * torch.tensor([2.0, 3.0])).sum().backward()
        close(x.grad, [-1.0, -1.5])
    R.test("GradReverse", gr)

    def train_ste():
        torch.manual_seed(0)
        w = torch.randn(4, requires_grad=True)
        target = torch.tensor([1.0, -2.0, 3.0, 0.0])
        opt = torch.optim.SGD([w], lr=0.1)
        for _ in range(200):
            loss = ((ns["RoundSTE"].apply(w) - target) ** 2).sum()
            opt.zero_grad(); loss.backward(); opt.step()
        close(torch.round(w.detach()), target, what="kuantize ağırlıklar hedefe ulaşmalı")
        return f"quantization-aware: round(w) = {torch.round(w.detach()).tolist()}"
    R.test("STE ile kuantize parametre öğrenme", train_ste)
