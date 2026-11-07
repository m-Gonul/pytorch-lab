import torch
from torch import nn
from labcheck import close, need, ensure


def deep(width=128, depth=20, act=nn.ReLU):
    L = []
    for _ in range(depth):
        L += [nn.Linear(width, width), act()]
    return nn.Sequential(*L)


def check(ns, R):
    def init():
        torch.manual_seed(0)
        m = need(ns, "init_model")(nn.Sequential(nn.Linear(500, 400), nn.ReLU(), nn.Linear(400, 300)), "kaiming")
        ensure(abs(m[0].weight.detach().std().item() / (2 / 500) ** 0.5 - 1) < 0.05, f"kaiming std {m[0].weight.detach().std().item():.4f}")
        ensure(float(m[0].bias.detach().abs().max()) == 0, "bias'lar 0 olmalı")
        m = need(ns, "init_model")(m, "xavier")
        ensure(abs(m[2].weight.detach().std().item() / (2 / 700) ** 0.5 - 1) < 0.05, f"xavier std {m[2].weight.detach().std().item():.4f}")
    R.test("init_model: doğru std'ler, sıfır bias", init)

    def stats():
        torch.manual_seed(0)
        X = torch.randn(256, 128)
        rows = []
        res = {}
        for scheme, act in (("kaiming", nn.ReLU), ("xavier", nn.ReLU), ("xavier", nn.Tanh)):
            torch.manual_seed(1)
            m = need(ns, "init_model")(deep(act=act), scheme)
            a, g = need(ns, "layer_stats")(m, X)
            ensure(len(a) == 20 and len(g) == 20, f"20 değer bekleniyordu, {len(a)} ve {len(g)} geldi")
            res[(scheme, act.__name__)] = (a, g)
            rows.append([f"{scheme} + {act.__name__}", f"{a[0]:.3g}", f"{a[-1]:.3g}", f"{g[0]:.3g}", f"{g[-1]:.3g}"])
        R.table("20 katman: aktivasyon ve gradyan std'si", ["kurulum", "akt. ilk", "akt. son", "grad ilk", "grad son"], rows)
        a, g = res[("kaiming", "ReLU")]
        ensure(0.2 < a[-1] / a[0] < 5, "Kaiming + ReLU aktivasyonları kararlı olmalı")
        a2, _ = res[("xavier", "ReLU")]
        ensure(a2[-1] < 0.05 * a2[0], "Xavier + ReLU aktivasyonları sönmeli")
    R.test("layer_stats: 20 katman, beklenen davranışlar", stats)

    def clean():
        torch.manual_seed(0)
        m = need(ns, "init_model")(deep(depth=3), "kaiming")
        need(ns, "layer_stats")(m, torch.randn(8, 128))
        ensure(all(len(x._forward_hooks) == 0 for x in m.modules()), "hook'lar kaldırılmamış")
    R.test("hook'lar sonunda kaldırılıyor", clean)
