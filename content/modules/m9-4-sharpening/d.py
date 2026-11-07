import math
import torch
import torch.nn.functional as F


def gaussian_kernel1d(sigma, ksize):
    ...


class UnsharpMask(torch.nn.Module):
    def __init__(self, ksize=9):
        super().__init__()
        ...

    def forward(self, x):
        ...


def fit_unsharp(x, target, steps=400, lr=0.05):
    ...


from skimage import data
x = torch.tensor(data.moon(), dtype=torch.float32)[:128, :128] / 255
print(UnsharpMask()(x).shape)
