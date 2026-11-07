import torch
from torch import nn
from labcheck import close, no_loops, need, ensure, bench, fmt_time


def check(ns, R):
    src = ns["__source__"]
    R.test("döngü yok", lambda: no_loops(src))
    g = torch.Generator().manual_seed(0)

    def mk(bias, stride=1):
        conv = nn.Conv2d(8, 16, 3, stride, 1, bias=bias).double()
        bn = nn.BatchNorm2d(16).double()
        with torch.no_grad():
            bn.running_mean.copy_(torch.randn(16, generator=g, dtype=torch.float64))
            bn.running_var.copy_(torch.rand(16, generator=g, dtype=torch.float64) + 0.5)
            bn.weight.copy_(torch.randn(16, generator=g, dtype=torch.float64))
            bn.bias.copy_(torch.randn(16, generator=g, dtype=torch.float64))
        conv.eval(); bn.eval()
        return conv, bn

    def eq():
        for bias, s in ((False, 1), (True, 2)):
            conv, bn = mk(bias, s)
            f = need(ns, "fuse_conv_bn")(conv, bn)
            ensure(isinstance(f, nn.Conv2d) and f.bias is not None, "bias'lı nn.Conv2d bekleniyordu")
            x = torch.randn(3, 8, 10, 10, generator=g, dtype=torch.float64)
            with torch.no_grad():
                close(f(x), bn(conv(x)), what=f"bias={bias}, stride={s}")
    R.test("fused conv = bn(conv(x)) (eval)", eq)
    if "fuse_conv_bn" not in ns:
        return
    conv, bn = nn.Conv2d(64, 64, 3, 1, 1, bias=False).eval(), nn.BatchNorm2d(64).eval()
    f = ns["fuse_conv_bn"](conv, bn)
    x = torch.randn(32, 64, 32, 32, generator=g)
    with torch.no_grad():
        t1, t2 = bench(lambda: bn(conv(x)), repeat=3), bench(lambda: f(x), repeat=3)
    R.table("Çıkarım, (32, 64, 32, 32), 3×3 conv", ["yöntem", "süre"],
            [["conv → BN (iki geçiş)", fmt_time(t1)], ["senin fused conv", fmt_time(t2)]],
            note=f"{t1 / t2:.2f}× hızlı: BN için tüm aktivasyon tensörünü bir kez daha okuyup yazmak gerekmiyor.")
