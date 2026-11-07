import torch
from torch import nn
from labcheck import close, max_loops, need, ensure, bench, fmt_time


def check(ns, R):
    src = ns["__source__"]
    R.test("ensemble fonksiyonlarında döngü yok", lambda: (max_loops(src, 0, func="ensemble_logits"), max_loops(src, 0, func="ensemble_predict")))
    torch.manual_seed(0)
    models = [nn.Sequential(nn.Linear(4, 24), nn.ReLU(), nn.Linear(24, 5)) for _ in range(8)]
    X = torch.randn(50, 4)
    ref = torch.stack([m(X) for m in models])

    def lg():
        out = need(ns, "ensemble_logits")(models, X)
        ensure(tuple(out.shape) == (8, 50, 5), f"shape {tuple(out.shape)}")
        close(out, ref, rtol=1e-5, atol=1e-6)
    R.test("ensemble_logits = modeller üzerinde döngü", lg)
    R.test("ensemble_predict: ortalama olasılık argmax", lambda: ensure(torch.equal(need(ns, "ensemble_predict")(models, X), ref.softmax(-1).mean(0).argmax(-1)), "tahminler farklı"))
    if "ensemble_logits" not in ns:
        return
    big = [nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10)) for _ in range(64)]
    Xb = torch.randn(256, 16)
    with torch.no_grad():
        t_loop = bench(lambda: torch.stack([m(Xb) for m in big]), repeat=3)
        t_vec = bench(lambda: ns["ensemble_logits"](big, Xb), repeat=3)
    R.table("64 MLP × 256 örnek, ileri geçiş", ["yöntem", "süre", "döngüye göre"],
            [["for m in models: m(X)", fmt_time(t_loop), "1×"], ["senin ensemble_logits (vmap)", fmt_time(t_vec), f"{t_loop / t_vec:,.1f}× hızlı"]],
            note="Süreye stack_module_state'in parametre kopyası da dahil; gerçek kullanımda bir kez yapılır.")
