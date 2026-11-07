import torch
from labcheck import close, need, ensure


def check(ns, R):
    def gn():
        x = torch.randn(4, requires_grad=True); w = torch.randn(4, requires_grad=True)
        h = x * w
        y = (h.exp() + h.sin() * x).sum()
        nodes = need(ns, "graph_nodes")(y)
        ensure(len(nodes) == len(set(nodes)), "tekrar eden düğüm var")
        ensure(nodes[0] is y.grad_fn, "ilk düğüm çıktının grad_fn'i")
        pos = {n: i for i, n in enumerate(nodes)}
        for n in nodes:
            for c, _ in n.next_functions:
                if c is not None:
                    ensure(c in pos and pos[n] < pos[c], f"{type(n).__name__}, {type(c).__name__}'den önce gelmeli")
        names = sorted(type(n).__name__ for n in nodes)
        ensure(names.count("AccumulateGrad") == 2, "iki yaprak → iki AccumulateGrad")
        return " → ".join(type(n).__name__ for n in nodes)
    R.test("graph_nodes: topolojik ve tam", gn)

    def ss():
        a, W = torch.randn(64, 128, requires_grad=True), torch.randn(128, 128, requires_grad=True)
        ensure(tuple(need(ns, "saved_stats")(lambda a, W: torch.tanh(a @ W), a, W)) == (3, 131072), "tanh(a @ W): 3 tensör, 131072 bayt")
        n, b = ns["saved_stats"](lambda a: (a * 2).sum(), a)
        ensure(n == 0, f"skaler ile çarpma ve sum hiçbir şey saklamamalı; {n} sayıldı")
        n, b = ns["saved_stats"](lambda a: a.exp().exp(), a)
        ensure(n == 2 and b == 2 * 64 * 128 * 4, "iki exp → iki çıktı saklanır")
    R.test("saved_stats", ss)

    def hk():
        x = torch.tensor([3.0, 4.0], requires_grad=True)
        h = x * 1.0
        need(ns, "clip_grad_hook")(h, 1.0)
        (h * torch.tensor([3.0, 4.0])).sum().backward()
        close(x.grad, [0.6, 0.8], what="norm 5 → 1'e kırpıldı")
        x.grad = None
        h = x * 1.0
        ns["clip_grad_hook"](h, 10.0)
        (h * torch.tensor([3.0, 4.0])).sum().backward()
        close(x.grad, [3.0, 4.0], what="eşiğin altında dokunulmaz")
    R.test("clip_grad_hook", hk)

    def handle():
        x = torch.ones(2, requires_grad=True)
        h = x * 1.0
        hd = need(ns, "clip_grad_hook")(h, 0.1)
        hd.remove()
        (h * 10).sum().backward()
        close(x.grad, [10.0, 10.0], what="kaldırılan hook etkisiz")
    R.test("handle.remove()", handle)
