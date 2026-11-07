import copy
import torch
from torch import nn


@torch.no_grad()
def fuse_conv_bn(conv, bn):
    """→ yeni nn.Conv2d; eval modunda bn(conv(x)) ile aynı. Döngü yok."""
    fused = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride,
                      conv.padding, conv.dilation, conv.groups, bias=True).to(conv.weight.dtype)
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)               # (F,)
    b = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
    fused.weight.copy_(conv.weight * scale[:, None, None, None])
    fused.bias.copy_((b - bn.running_mean) * scale + bn.bias)
    return fused
