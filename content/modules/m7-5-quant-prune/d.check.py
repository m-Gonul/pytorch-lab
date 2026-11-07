import copy
import torch
import torch.nn.functional as F
from torch import nn
from labcheck import close, need, ensure


def check(ns, R):
    torch.manual_seed(0)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
            self.head = nn.Linear(256, 10)

        def forward(self, x):
            return self.head(self.body(x))
    m = Net()
    x = torch.randn(64, 128)

    def ql():
        L = nn.Linear(16, 8)
        Q = need(ns, "QuantLinear", "sınıf").from_float(L)
        ensure(Q.weight_int8.dtype == torch.int8 and Q.scale.shape == (8,), "int8 ağırlık, (out,) ölçek")
        ensure(int(Q.weight_int8.abs().amax(1).min()) == 127, "kanal başına ölçek: her satırın max'ı 127")
        W = Q.weight_int8.float() * Q.scale[:, None]
        ensure(((W - L.weight).abs() <= Q.scale[:, None] / 2 + 1e-7).all(), "hata ≤ s/2")
    R.test("QuantLinear.from_float", ql)

    def qm():
        q = need(ns, "quantize_model")(m)
        ensure(not any(isinstance(mod, nn.Linear) for mod in q.modules()), "iç içe tüm Linear'lar değişmeli")
        ensure(any(isinstance(mod, nn.Linear) for mod in m.modules()), "orijinal model değişmemeli (deepcopy)")
        with torch.no_grad():
            y, yq = m(x), q(x)
        rel = ((yq - y).norm() / y.norm()).item()
        ensure(rel < 0.02, f"göreli hata {rel:.3e}")
        b0, b1 = need(ns, "state_bytes")(m), ns["state_bytes"](q)
        ensure(b1 < 0.3 * b0, f"boyut {b0:,} → {b1:,}")
        return f"göreli hata {rel:.2e}; state_dict {b0:,} → {b1:,} bayt ({b0 / b1:.1f}×)"
    R.test("quantize_model: doğruluk ve boyut", qm)

    def pr():
        mm = copy.deepcopy(m)
        need(ns, "prune_global")(mm, 0.6)
        ws = [mod.weight for mod in mm.modules() if isinstance(mod, nn.Linear)]
        tot = sum(w.numel() for w in ws); zeros = sum(int((w == 0).sum()) for w in ws)
        ensure(abs(zeros / tot - 0.6) < 0.01, f"seyreklik {zeros / tot:.3f}")
        ensure(not any(n.endswith("_orig") for n, _ in mm.named_parameters()), "prune.remove çağrılmalı (weight_orig kalmamalı)")
        per = [f"{float((w == 0).float().mean()):.2f}" for w in ws]
        return f"global %60: katman başına seyreklik {per} (global eşik katmanlara farklı oranlar düşürür)"
    R.test("prune_global", pr)
