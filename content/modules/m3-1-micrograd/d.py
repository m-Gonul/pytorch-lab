import torch


def karpathy_grads():
    """L = (a·b + c)·f için {'a','b','c','f'} → gradyan."""
    ...


def neuron_torch(x, w, b):
    """o = tanh(x·w + b). → (o, ∂o/∂w, ∂o/∂x)"""
    x = x.detach().clone().requires_grad_(True)
    w = w.detach().clone().requires_grad_(True)
    ...


print(karpathy_grads())
print(neuron_torch(torch.tensor([2.0, 0.0]), torch.tensor([-3.0, 1.0]), 6.8813735870195432))
