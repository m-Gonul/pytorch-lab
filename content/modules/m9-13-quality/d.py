import math
import torch
import torch.nn.functional as F


def ssim_t(x, y, R=255.0):
    ...


def mse_sphere(ref, mse, sign, steps=200, lr=2.0, seed=0):
    ...


x = torch.rand(64, 64, dtype=torch.float64) * 255
print(ssim_t(x, x))
