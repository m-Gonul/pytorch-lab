import torch
from torch import nn


def init_model(model, scheme):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if scheme == "kaiming":
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif scheme == "xavier":
                nn.init.xavier_normal_(m.weight)
            else:
                raise ValueError(scheme)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return model


def layer_stats(model, X):
    """→ (act_stds, grad_stds): her nn.Linear çıktısı için, katman sırasıyla."""
    acts, grads, handles = [], {}, []
    for i, m in enumerate(mm for mm in model.modules() if isinstance(mm, nn.Linear)):
        def fwd(mod, inp, out, i=i):
            acts.append(out.detach().std().item())
            out.register_hook(lambda g, i=i: grads.__setitem__(i, g.std().item()))
        handles.append(m.register_forward_hook(fwd))
    model(X).pow(2).mean().backward()
    for h in handles:
        h.remove()
    return acts, [grads[i] for i in range(len(acts))]


def deep_mlp(width=128, depth=20, act=nn.ReLU):
    layers = []
    for _ in range(depth):
        layers += [nn.Linear(width, width), act()]
    return nn.Sequential(*layers)
