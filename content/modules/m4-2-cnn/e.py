import copy
import torch
from torch import nn


@torch.no_grad()
def fuse_conv_bn(conv, bn):
    """→ yeni nn.Conv2d; eval modunda bn(conv(x)) ile aynı. Döngü yok."""
    ...
