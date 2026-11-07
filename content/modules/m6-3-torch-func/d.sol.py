import torch
import torch.nn.functional as F
from torch.func import functional_call, vmap, grad, jacrev, jacfwd, jvp


def per_sample_grads(model, x, y):
    params = {k: v.detach() for k, v in model.named_parameters()}

    def loss_one(p, xi, yi):
        out = functional_call(model, p, (xi[None],))
        return F.cross_entropy(out, yi[None])
    return vmap(grad(loss_one), in_dims=(None, 0, 0))(params, x, y)


def jacobians(f, x):
    return jacrev(f)(x), jacfwd(f)(x)


def hvp(f, x, v):
    return jvp(grad(f), (x,), (v,))[1]


torch.manual_seed(0)
model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Tanh(), torch.nn.Linear(8, 3))
g = per_sample_grads(model, torch.randn(5, 4), torch.randint(0, 3, (5,)))
print({k: tuple(v.shape) for k, v in g.items()})
