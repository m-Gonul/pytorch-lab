import torch
import torch.nn.functional as F
from torch.func import functional_call, vmap, grad, jacrev, jacfwd, jvp


def per_sample_grads(model, x, y):
    ...


def jacobians(f, x):
    ...


def hvp(f, x, v):
    ...


torch.manual_seed(0)
model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Tanh(), torch.nn.Linear(8, 3))
g = per_sample_grads(model, torch.randn(5, 4), torch.randint(0, 3, (5,)))
print({k: tuple(v.shape) for k, v in g.items()})
