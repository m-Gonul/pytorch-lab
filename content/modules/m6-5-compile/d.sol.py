import torch
import torch.fx as fx
from torch import nn


def fold_conv_bn(conv, bn):
    fused = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                      conv.dilation, conv.groups, bias=True, padding_mode=conv.padding_mode,
                      device=conv.weight.device, dtype=conv.weight.dtype)
    s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    b = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
    with torch.no_grad():
        fused.weight.copy_(conv.weight * s[:, None, None, None])
        fused.bias.copy_((b - bn.running_mean) * s + bn.bias)
    return fused


def fuse_model(model):
    gm = fx.symbolic_trace(model)
    mods = dict(gm.named_modules())
    for node in list(gm.graph.nodes):
        if node.op != "call_module" or not isinstance(mods[node.target], nn.BatchNorm2d):
            continue
        prev = node.args[0]
        if not (isinstance(prev, fx.Node) and prev.op == "call_module" and isinstance(mods[prev.target], nn.Conv2d)):
            continue
        if len(prev.users) > 1:
            continue
        fused = fold_conv_bn(mods[prev.target], mods[node.target])
        parent, _, attr = prev.target.rpartition(".")
        setattr(gm.get_submodule(parent) if parent else gm, attr, fused)
        node.replace_all_uses_with(prev)
        gm.graph.erase_node(node)
    gm.graph.lint()
    gm.delete_all_unused_submodules()
    gm.recompile()
    return gm


torch.manual_seed(0)
m = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.Conv2d(8, 8, 3, bias=False), nn.BatchNorm2d(8)).eval()
fm = fuse_model(m)
x = torch.randn(2, 3, 16, 16)
print((fm(x) - m(x)).abs().max(), [type(mod).__name__ for mod in fm.modules()])
