import math
import torch
import torch.nn.functional as F


def warp(img, flow):
    ...


def lk_level(I0, I1, flow, win=7, iters=5):
    ...


def pyr_lk(I0, I1, levels=4, win=7, iters=5):
    ...


def fb_error(f01, f10):
    ...


def fruc_mc(I0, I2, flow02):
    ...


x = torch.rand(32, 32, dtype=torch.float64)
print(warp(x, torch.zeros(2, 32, 32, dtype=torch.float64)).allclose(x))
