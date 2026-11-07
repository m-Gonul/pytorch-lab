import copy
import torch
from torch import nn
from torch.func import stack_module_state, functional_call


def ensemble_logits(models, X):
    """→ (E, N, C). Döngü yok."""
    params, buffers = stack_module_state(models)
    base = copy.deepcopy(models[0]).to("meta")
    f = lambda p, b, x: functional_call(base, (p, b), (x,))
    return torch.vmap(f, in_dims=(0, 0, None))(params, buffers, X)


def ensemble_predict(models, X):
    """→ (N,): ortalama olasılığın argmax'ı. Döngü yok."""
    return ensemble_logits(models, X).softmax(-1).mean(0).argmax(-1)


torch.manual_seed(0)
models = [nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 3)) for _ in range(16)]
X = torch.randn(100, 2)
print(ensemble_logits(models, X).shape)    # (16, 100, 3)
