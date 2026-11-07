import torch
import torch.nn.functional as F


def di_zenzo(img):
    ...


def channel_sum_grad(img):
    ...


def mahalanobis_mask(img, samples, D0):
    ...


def box_mask(img, samples, k=1.25):
    ...


x = torch.rand(1, 3, 16, 16)
print(di_zenzo(x)[0].shape)
