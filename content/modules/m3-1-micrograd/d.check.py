import torch
from labcheck import close, need, ensure


def check(ns, R):
    def kg():
        g = need(ns, "karpathy_grads")()
        ensure(isinstance(g, dict) and set(g) >= {"a", "b", "c", "f"}, "{'a','b','c','f'} anahtarlı sözlük bekleniyordu")
        close([float(g[k]) for k in "abcf"], [6, -4, -2, 4])
    R.test("karpathy_grads: elle bulduğunla aynı", kg)

    def nt():
        out = need(ns, "neuron_torch")(torch.tensor([2.0, 0.0], dtype=torch.float64), torch.tensor([-3.0, 1.0], dtype=torch.float64), 6.8813735870195432)
        ensure(isinstance(out, (tuple, list)) and len(out) == 3, "(o, dw, dx) döndürmeli")
        o, dw, dx = out
        close(o, 2 ** -0.5); close(dw, [1.0, 0.0], atol=1e-9); close(dx, [-1.5, 0.5])
    R.test("neuron_torch: ∂w = [1, 0], ∂x = [−1.5, 0.5]", nt)

    def graph():
        a = torch.tensor(2.0, requires_grad=True); b = torch.tensor(-3.0, requires_grad=True)
        L = (a * b + 10) * -2
        names, fn = [], L.grad_fn
        while fn is not None:
            names.append(type(fn).__name__)
            nxt = [f for f, _ in fn.next_functions if f is not None and type(f).__name__ != "AccumulateGrad"]
            fn = nxt[0] if nxt else None
        return "PyTorch'un grafiği (çıktıdan geriye): " + " → ".join(names)
    R.test("PyTorch'un hesap grafiği", graph)
