import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

T = 200
BETAS = torch.linspace(1e-4, 0.05, T)
AB = torch.cumprod(1 - BETAS, 0)


def cfg_two_calls(net, x, t, y, s):
    ...


def cfg_batched(net, x, t, y, s):
    ...


@torch.no_grad()
def ddpm_sample(net, labels, s, seed):
    ...


@torch.no_grad()
def ddim_sample(net, labels, s, steps, seed):
    ...
