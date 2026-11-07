import itertools
import torch
import torch.nn.functional as F


def nlm(img, h, tw=7, sw=21, sigma=0.0):
    ...


print(nlm(torch.rand(24, 24) * 255, 10.0, 3, 5).shape)
