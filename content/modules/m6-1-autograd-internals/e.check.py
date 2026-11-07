import torch
from labcheck import close, need, ensure


def saved_bytes(fn, *args):
    tot = [0]

    def outer_pack(t):
        tot[0] += t.numel() * t.element_size()
        return t
    with torch.autograd.graph.saved_tensors_hooks(outer_pack, lambda t: t):
        out = fn(*args)
    return out, tot[0]


def check(ns, R):
    torch.manual_seed(0)
    mlp = torch.nn.Sequential(torch.nn.Linear(256, 512), torch.nn.GELU(), torch.nn.Linear(512, 512), torch.nn.GELU(), torch.nn.Linear(512, 10))
    x = torch.randn(256, 256)

    def fwd():
        y_ref = mlp(x)
        y = need(ns, "bf16_saved")(mlp, x)
        close(y, y_ref, what="ileri geçiş değişmemeli")
    R.test("ileri geçiş aynı", fwd)

    def grads():
        mlp.zero_grad(); mlp(x).pow(2).mean().backward()
        ref = [p.grad.clone() for p in mlp.parameters()]
        mlp.zero_grad(); need(ns, "bf16_saved")(mlp, x).pow(2).mean().backward()
        rel = max(((p.grad - r).norm() / r.norm()).item() for p, r in zip(mlp.parameters(), ref))
        ensure(0 < rel < 2e-2, f"en büyük göreli gradyan hatası {rel:.2e}")
        return f"en büyük göreli gradyan hatası {rel:.2e}"
    R.test("gradyanlar yakın (ama birebir değil)", grads)

    def mem():
        _, full = saved_bytes(mlp, x)
        tot = [0]

        def count(fn):
            orig = torch.autograd.graph.saved_tensors_hooks

            class Wrap(orig):
                def __init__(self, pack, unpack):
                    def p(t):
                        r = pack(t)
                        flat = r if isinstance(r, (tuple, list)) else (r,)
                        tot[0] += sum(v.numel() * v.element_size() for v in flat if isinstance(v, torch.Tensor))
                        return r
                    super().__init__(p, unpack)
            torch.autograd.graph.saved_tensors_hooks = Wrap
            try:
                fn()
            finally:
                torch.autograd.graph.saved_tensors_hooks = orig
        count(lambda: need(ns, "bf16_saved")(mlp, x))
        ensure(tot[0] > 0, "saved_tensors_hooks kullanılmalı")
        ensure(abs(tot[0] / full - 0.5) < 0.02, f"saklanan bayt oranı {tot[0] / full:.3f}; ~0.5 bekleniyordu")
        R.table("Saklanan tensörler (5 katmanlı MLP, B = 256)", ["", "bayt"],
                [["fp32", f"{full:,}"], ["bf16 paket", f"{tot[0]:,}"]],
                note="Not: parametreler de (matmul girdisi olarak) saklananlar arasında sayılır; pratikte yalnız aktivasyonları paketlemek için boyuta göre filtrelenir.")
    R.test("saklanan bellek yarıya iniyor", mem)
