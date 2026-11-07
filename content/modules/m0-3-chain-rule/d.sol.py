import torch


def grads_autograd(x, y):
    """f = (x + y) * y için (df/dx, df/dy) — backward ile."""
    x = torch.tensor(float(x), requires_grad=True)
    y = torch.tensor(float(y), requires_grad=True)
    f = (x + y) * y
    f.backward()
    return x.grad, y.grad


def g(p):
    x, y = p[0], p[1]
    return torch.stack([x * y, x + y ** 2])


def jacobian_torch(p):
    """p = tensor([x, y]) → g(p) = [x*y, x + y**2]'nin (2, 2) Jacobian'ı."""
    return torch.func.jacrev(g)(p)


print(grads_autograd(2.0, 3.0))                        # (tensor(3.), tensor(8.))
print(jacobian_torch(torch.tensor([2.0, 3.0])))
