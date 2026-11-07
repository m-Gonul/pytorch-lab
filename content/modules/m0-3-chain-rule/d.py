import torch


def grads_autograd(x, y):
    """f = (x + y) * y için (df/dx, df/dy) — backward ile."""
    ...


def jacobian_torch(p):
    """p = tensor([x, y]) → g(p) = [x*y, x + y**2]'nin (2, 2) Jacobian'ı."""
    ...


print(grads_autograd(2.0, 3.0))                        # (tensor(3.), tensor(8.))
print(jacobian_torch(torch.tensor([2.0, 3.0])))
