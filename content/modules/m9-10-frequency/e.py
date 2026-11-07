import torch
import torch.nn.functional as F


def conv2d_fft(x, w, padding=0):
    ...


x = torch.randn(1, 2, 16, 16); w = torch.randn(3, 2, 5, 5)
print((conv2d_fft(x, w, 2) - F.conv2d(x, w, padding=2)).abs().max())
