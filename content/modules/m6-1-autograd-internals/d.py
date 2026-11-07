import torch


def graph_nodes(t):
    ...


def saved_stats(fn, *args):
    ...


def clip_grad_hook(t, max_norm):
    ...


x = torch.randn(3, requires_grad=True); w = torch.randn(3, requires_grad=True)
y = (x * w).exp().sum()
print([type(n).__name__ for n in graph_nodes(y)])
a, W = torch.randn(64, 128, requires_grad=True), torch.randn(128, 128, requires_grad=True)
print(saved_stats(lambda a, W: torch.tanh(a @ W), a, W))   # (3, 131072)
