import torch
import torch.nn.functional as F


def running_max(x, k):
    ...


def rect_dilate(img, kh, kw):
    ...


def rect_erode(img, kh, kw):
    ...


print(running_max(torch.tensor([1.0, 5, 2, 0, 3, 9, 1]), 3))
