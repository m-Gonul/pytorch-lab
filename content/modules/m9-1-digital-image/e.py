import torch
import torch.nn.functional as F


def warp_batch(imgs, A, t):
    """imgs (B, H, W), A (B, 2, 2), t (B, 2). Döngü yok."""
    ...


imgs = torch.rand(4, 32, 32)
print(warp_batch(imgs, torch.eye(2).repeat(4, 1, 1), torch.zeros(4, 2)).shape)
