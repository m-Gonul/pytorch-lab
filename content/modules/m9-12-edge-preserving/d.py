import math
import torch
import torch.nn.functional as F


def bilateral_t(img, r, sc, ss):
    ...


def fit_bilateral(noisy, clean, r=3, steps=150, lr=0.05):
    ...


x = torch.rand(16, 16) * 255
print(bilateral_t(x, 2, 30.0, 2.0).shape)
