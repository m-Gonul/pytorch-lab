import math
import torch
import torch.nn.functional as F


def dog_pyramid(img, octaves=4, s=3, sigma0=1.6):
    ...


def detect(img, octaves=4, s=3, sigma0=1.6, thr=0.03):
    ...


x = torch.zeros(64, 64, dtype=torch.float64); x[28:36, 28:36] = 1
print(len(detect(x, octaves=2)))
