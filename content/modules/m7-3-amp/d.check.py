import copy
import torch
import torch.nn.functional as F
from labcheck import close, need, ensure


def check(ns, R):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float16 if dev == "cuda" else torch.bfloat16
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(20, 64), torch.nn.ReLU(), torch.nn.Linear(64, 3)).to(dev)
    x, y = torch.randn(256, 20, device=dev), torch.randint(0, 3, (256,), device=dev)

    def dts():
        d = need(ns, "amp_dtypes")(model, x, y, dev, dt)
        ensure(d["linear"] == dt, f"Linear çıktısı {d['linear']}; autocast altında {dt} bekleniyordu")
        ensure(d["loss"] == torch.float32, f"cross_entropy fp32'de hesaplanmalı ({d['loss']})")
        ensure(d["param"] == torch.float32 and d["grad"] == torch.float32, "parametre ve gradyan fp32 kalmalı")
        return ", ".join(f"{k}: {str(v).replace('torch.', '')}" for k, v in d.items())
    R.test("autocast dtype'ları", dts)

    def train():
        m = copy.deepcopy(model)
        opt = torch.optim.Adam(m.parameters(), lr=1e-2)
        scaler = torch.amp.GradScaler(dev)
        first = need(ns, "train_step_amp")(m, opt, scaler, x, y, dev, dt)
        for _ in range(150):
            last = ns["train_step_amp"](m, opt, scaler, x, y, dev, dt)
        ensure(last < 0.5 * first, f"kayıp {first:.3f} → {last:.3f}")
        return f"kayıp {first:.3f} → {last:.3f}"
    R.test("AMP ile eğitim", train)

    def skip():
        m = copy.deepcopy(model)
        opt = torch.optim.SGD(m.parameters(), lr=0.1)
        scaler = torch.amp.GradScaler(dev, init_scale=1024.0)
        before = [p.detach().clone() for p in m.parameters()]
        h = m[0].weight.register_hook(lambda g: g * float("inf"))
        need(ns, "train_step_amp")(m, opt, scaler, x, y, dev, dt)
        h.remove()
        for p, q in zip(m.parameters(), before):
            close(p, q, what="inf gradyanlı adım atlanmalı")
        ensure(scaler.get_scale() < 1024.0, f"ölçek küçülmeliydi ({scaler.get_scale()})")
    R.test("inf gradyan: adım atlanıyor, ölçek küçülüyor", skip)
