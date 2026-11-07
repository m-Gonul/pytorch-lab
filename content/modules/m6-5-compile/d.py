import torch
import torch.fx as fx
from torch import nn


def fold_conv_bn(conv, bn):
    ...


def fuse_model(model):
    ...


torch.manual_seed(0)
m = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.Conv2d(8, 8, 3, bias=False), nn.BatchNorm2d(8)).eval()
fm = fuse_model(m)
x = torch.randn(2, 3, 16, 16)
print((fm(x) - m(x)).abs().max(), [type(mod).__name__ for mod in fm.modules()])
