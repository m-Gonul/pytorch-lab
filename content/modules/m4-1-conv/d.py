import torch
import torch.nn.functional as F


def conv2d_unfold(x, w, b, stride=1, pad=0):
    ...


def conv2d_input_grad(dout, w, x_shape, stride=1, pad=0):
    """dx, autograd kullanmadan (F.fold ile)."""
    ...
