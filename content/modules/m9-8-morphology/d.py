import numpy as np
import torch
import torch.nn.functional as F


def dilate_t(f, w):
    ...


def learn_se(img, target_se, steps=1500, lr=0.02, patch=128, seed=0):
    ...


def tophat_t(f, k):
    ...


f = torch.rand(1, 1, 8, 8)
print(dilate_t(f, torch.zeros(3, 3)).shape)
