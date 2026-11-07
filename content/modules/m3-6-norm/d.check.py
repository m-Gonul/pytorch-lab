import torch
from torch import nn
from labcheck import close, need, ensure


def check(ns, R):
    g = torch.Generator().manual_seed(0)

    def bn():
        C = need(ns, "MyBatchNorm1d", "sınıf")
        a, b = C(6).double(), nn.BatchNorm1d(6).double()
        with torch.no_grad():
            w, bb = torch.rand(6, generator=g, dtype=torch.float64) + 0.5, torch.randn(6, generator=g, dtype=torch.float64)
            a.weight.copy_(w); a.bias.copy_(bb); b.weight.copy_(w); b.bias.copy_(bb)
        ensure(set(dict(a.named_buffers())) >= {"running_mean", "running_var"}, "running_mean/var buffer olarak kayıtlı olmalı")
        for _ in range(5):
            x = torch.randn(16, 6, generator=g, dtype=torch.float64) * 3 + 1
            xa, xb = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
            ya, yb = a(xa), b(xb)
            close(ya, yb, what="eğitim çıktısı")
            w = torch.randn(16, 6, generator=g, dtype=torch.float64)
            (ya * w).sum().backward(); (yb * w).sum().backward()
            close(xa.grad, xb.grad, what="dx")
        close(a.running_mean, b.running_mean, what="running_mean"); close(a.running_var, b.running_var, what="running_var (yansız!)")
        a.eval(); b.eval()
        x = torch.randn(4, 6, generator=g, dtype=torch.float64)
        close(a(x), b(x), what="eval çıktısı")
        close(a.weight.grad, b.weight.grad, what="dγ")
    R.test("MyBatchNorm1d = nn.BatchNorm1d (5 adım, train + eval)", bn)

    def rms():
        C = need(ns, "MyRMSNorm", "sınıf")
        a, b = C(8, eps=1e-6).double(), nn.RMSNorm(8, eps=1e-6).double()
        with torch.no_grad():
            w = torch.rand(8, generator=g, dtype=torch.float64) + 0.5
            a.weight.copy_(w); b.weight.copy_(w)
        x = torch.randn(3, 5, 8, generator=g, dtype=torch.float64)
        xa, xb = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
        ya, yb = a(xa), b(xb)
        close(ya, yb)
        ya.sum().backward(); yb.sum().backward()
        close(xa.grad, xb.grad, what="dx")
    R.test("MyRMSNorm = nn.RMSNorm", rms)
