import torch
import torch.nn.functional as F


def conv2d_strided(x, w, b, stride=1, pad=0):
    """F.pad + iki .unfold (view) + einsum. Döngü yok."""
    KH, KW = w.shape[2:]
    xp = F.pad(x, (pad, pad, pad, pad))
    win = xp.unfold(2, KH, stride).unfold(3, KW, stride)        # (N, C, OH, OW, KH, KW) view
    return torch.einsum("nchwkl,fckl->nfhw", win, w) + b[None, :, None, None]
