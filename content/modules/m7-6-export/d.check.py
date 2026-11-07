import logging
import torch
from torch.export import export
from labcheck import close, need, ensure

torch._logging.set_logs(dynamo=logging.ERROR, export=logging.ERROR)


def check(ns, R):
    torch.manual_seed(0)
    m = need(ns, "MLP", "sınıf")().eval()

    def dyn():
        ep = need(ns, "export_dynamic")(m, torch.randn(4, 16))
        for B in (1, 7, 300):
            x = torch.randn(B, 16)
            close(ep.module()(x), m(x), what=f"batch {B}")
        names = need(ns, "op_names")(ep)
        ensure(sum("linear" in n for n in names) == 2 and any("relu" in n for n in names), f"işlemler {names}")
        return "ATen işlemleri: " + ", ".join(names)
    R.test("dinamik batch ile export", dyn)

    def sl():
        ep = ns["export_dynamic"](m, torch.randn(4, 16))
        ep2 = need(ns, "save_load")(ep)
        x = torch.randn(5, 16)
        close(ep2.module()(x), m(x))
    R.test("save/load (.pt2) gidiş-dönüş", sl)

    def bad():
        try:
            export(need(ns, "BadModel", "sınıf")(), (torch.randn(3),))
        except Exception as e:
            return f"Python if → {type(e).__name__}"
        raise AssertionError("BadModel export'u hata vermeliydi")
    R.test("veriye bağlı if export'ta reddediliyor", bad)

    def cond():
        ep = export(need(ns, "CondModel", "sınıf")(), (torch.randn(3),))
        for v in (torch.tensor([1.0, 2.0, 3.0]), torch.tensor([-1.0, -1.0, 0.5])):
            want = v * 2 if v.sum() > 0 else -v
            close(ep.module()(v), want, what=f"girdi {v.tolist()}")
        ensure(any("cond" in str(n.target) for n in ep.graph.nodes if n.op == "call_function"), "grafta cond düğümü olmalı")
    R.test("torch.cond: iki dal da grafta", cond)
