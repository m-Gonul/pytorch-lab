import os
import time
import hashlib
import numpy as np
import torch
from torch.utils.cpp_extension import load_inline


def mandel_torch(H, W, maxit):
    ...


def scaling(ext, H, W, maxit, threads):
    ...


def fit_amdahl(p, T):
    ...


print(mandel_torch(50, 75, 100).float().mean())
