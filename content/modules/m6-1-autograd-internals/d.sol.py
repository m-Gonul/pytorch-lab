import torch


def graph_nodes(t):
    order, seen = [], set()

    def dfs(fn):
        if fn is None or fn in seen:
            return
        seen.add(fn)
        for nxt, _ in fn.next_functions:
            dfs(nxt)
        order.append(fn)
    dfs(t.grad_fn)
    return order[::-1]


def saved_stats(fn, *args):
    stats = [0, 0]

    def pack(t):
        stats[0] += 1
        stats[1] += t.numel() * t.element_size()
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        fn(*args)
    return stats[0], stats[1]


def clip_grad_hook(t, max_norm):
    def hook(g):
        n = g.norm()
        return g * (max_norm / n) if n > max_norm else g
    return t.register_hook(hook)


x = torch.randn(3, requires_grad=True); w = torch.randn(3, requires_grad=True)
y = (x * w).exp().sum()
print([type(n).__name__ for n in graph_nodes(y)])
a, W = torch.randn(64, 128, requires_grad=True), torch.randn(128, 128, requires_grad=True)
print(saved_stats(lambda a, W: torch.tanh(a @ W), a, W))   # (3, 131072)
