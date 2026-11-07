import torch
import torch.nn.functional as F


def conv2d_unfold(x, w, b, stride=1, pad=0):
    N, C, H, W = x.shape
    Fo, _, KH, KW = w.shape
    OH, OW = (H + 2 * pad - KH) // stride + 1, (W + 2 * pad - KW) // stride + 1
    cols = F.unfold(x, (KH, KW), padding=pad, stride=stride)          # (N, C·KH·KW, L)
    out = w.reshape(Fo, -1) @ cols + b[:, None]                       # (N, F, L)
    return out.reshape(N, Fo, OH, OW)


def conv2d_input_grad(dout, w, x_shape, stride=1, pad=0):
    """dx, autograd kullanmadan (F.fold ile)."""
    N, Fo, OH, OW = dout.shape
    _, C, KH, KW = w.shape
    cols = w.reshape(Fo, -1).T @ dout.reshape(N, Fo, OH * OW)        # (N, C·KH·KW, L)
    return F.fold(cols, x_shape[2:], (KH, KW), padding=pad, stride=stride)
