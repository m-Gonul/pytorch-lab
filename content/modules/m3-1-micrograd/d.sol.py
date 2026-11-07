import torch


def karpathy_grads():
    """L = (a·b + c)·f için {'a','b','c','f'} → gradyan."""
    v = {k: torch.tensor(x, dtype=torch.float64, requires_grad=True) for k, x in dict(a=2.0, b=-3.0, c=10.0, f=-2.0).items()}
    L = (v["a"] * v["b"] + v["c"]) * v["f"]
    L.backward()
    return {k: t.grad.item() for k, t in v.items()}


def neuron_torch(x, w, b):
    """o = tanh(x·w + b). → (o, ∂o/∂w, ∂o/∂x)"""
    x = x.detach().clone().requires_grad_(True)
    w = w.detach().clone().requires_grad_(True)
    o = torch.tanh(x @ w + b)
    o.backward()
    return o.detach(), w.grad, x.grad


print(karpathy_grads())
print(neuron_torch(torch.tensor([2.0, 0.0]), torch.tensor([-3.0, 1.0]), 6.8813735870195432))
