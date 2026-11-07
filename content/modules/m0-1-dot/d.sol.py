import torch


def cosine_torch(u, v):
    return (u @ v) / (u.norm() * v.norm())


def cosine_grad(u, v):
    """∂cos(u, v)/∂u — autograd ile."""
    u = u.detach().clone().requires_grad_(True)
    c = cosine_torch(u, v)
    c.backward()
    return u.grad


u = torch.tensor([1., 2., 3.])
v = torch.tensor([4., -5., 6.])
print(cosine_torch(u, v))
print(cosine_grad(u, v))
