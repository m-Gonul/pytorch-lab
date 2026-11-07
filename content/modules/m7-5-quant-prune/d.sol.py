import copy
import torch
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from torch import nn


class QuantLinear(nn.Module):
    @classmethod
    def from_float(cls, linear):
        self = cls()
        W = linear.weight.detach()
        s = W.abs().amax(1).clamp_min(1e-12) / 127
        self.register_buffer("weight_int8", torch.clamp(torch.round(W / s[:, None]), -127, 127).to(torch.int8))
        self.register_buffer("scale", s.float())
        self.bias = None if linear.bias is None else nn.Parameter(linear.bias.detach().clone())
        return self

    def forward(self, x):
        return F.linear(x, self.weight_int8.float() * self.scale[:, None], self.bias)


def quantize_model(model):
    model = copy.deepcopy(model)

    def swap(mod):
        for name, child in mod.named_children():
            if isinstance(child, nn.Linear):
                setattr(mod, name, QuantLinear.from_float(child))
            else:
                swap(child)
    swap(model)
    return model


def state_bytes(model):
    return sum(t.numel() * t.element_size() for t in model.state_dict().values())


def prune_global(model, amount):
    params = [(m, "weight") for m in model.modules() if isinstance(m, nn.Linear)]
    prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=amount)
    for m, n in params:
        prune.remove(m, n)
    return model


torch.manual_seed(0)
m = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 10))
q = quantize_model(m)
x = torch.randn(4, 128)
print((q(x) - m(x)).abs().max(), state_bytes(m), state_bytes(q))
