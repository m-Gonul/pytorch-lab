import torch
from torch import nn
from labcheck import close, need, ensure


def randomize_bn(m):
    g = torch.Generator().manual_seed(1)
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.running_mean.copy_(torch.randn(mod.num_features, generator=g))
            mod.running_var.copy_(torch.rand(mod.num_features, generator=g) + 0.5)
            with torch.no_grad():
                mod.weight.copy_(torch.randn(mod.num_features, generator=g))
                mod.bias.copy_(torch.randn(mod.num_features, generator=g))


def check(ns, R):
    def fold():
        conv = nn.Conv2d(1, 1, 1); bn = nn.BatchNorm2d(1, eps=0.0).eval()
        with torch.no_grad():
            conv.weight.fill_(2.0); conv.bias.fill_(1.0)
            bn.running_mean.fill_(3.0); bn.running_var.fill_(4.0); bn.weight.fill_(0.5); bn.bias.fill_(1.0)
        f = need(ns, "fold_conv_bn")(conv, bn)
        close([f.weight.item(), f.bias.item()], [0.5, 0.5], what="elle örnek")
    R.test("fold_conv_bn (elle örnek)", fold)

    def model():
        torch.manual_seed(0)
        m = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
                          nn.Conv2d(8, 8, 3, stride=2, bias=False), nn.BatchNorm2d(8), nn.ReLU(), nn.BatchNorm2d(8)).double()
        randomize_bn(m); m.eval()
        x = torch.randn(4, 3, 20, 20, dtype=torch.float64)
        fm = need(ns, "fuse_model")(m)
        close(fm(x), m(x))
        n_bn = sum(isinstance(mod, nn.BatchNorm2d) for mod in fm.modules())
        ensure(n_bn == 1, f"iki BN katlanmalı, ReLU'dan sonraki BN kalmalı; kalan BN: {n_bn}")
        calls = [n.target for n in fm.graph.nodes if n.op == "call_module"]
        return f"FX grafı: {calls}"
    R.test("fuse_model: eval çıktısı aynı, 2 BN katlandı", model)

    def shared():
        class Two(nn.Module):
            def __init__(self):
                super().__init__(); self.c = nn.Conv2d(2, 2, 1); self.bn = nn.BatchNorm2d(2)

            def forward(self, x):
                y = self.c(x)
                return self.bn(y) + y
        torch.manual_seed(0)
        m = Two().double(); randomize_bn(m); m.eval()
        x = torch.randn(1, 2, 4, 4, dtype=torch.float64)
        close(need(ns, "fuse_model")(m)(x), m(x), what="conv çıktısı başka yerde de kullanılıyorsa katlama yapılmamalı")
    R.test("çok kullanıcılı conv'a dokunulmuyor", shared)
