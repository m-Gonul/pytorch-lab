import time
import torch
import torch.nn.functional as F


def patch_embed_unfold(x, W, b, P):
    return F.unfold(x, P, stride=P).transpose(1, 2) @ W.T + b


def patch_embed_reshape(x, W, b, P):
    B, C, H, Wd = x.shape
    z = x.view(B, C, H // P, P, Wd // P, P).permute(0, 2, 4, 1, 3, 5).reshape(B, (H // P) * (Wd // P), C * P * P)
    return z @ W.T + b
