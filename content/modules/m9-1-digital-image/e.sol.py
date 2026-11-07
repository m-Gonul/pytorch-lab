import torch
import torch.nn.functional as F


def warp_batch(imgs, A, t):
    """imgs (B, H, W), A (B, 2, 2), t (B, 2). Döngü yok."""
    B, H, W = imgs.shape
    ys, xs = torch.meshgrid(torch.arange(H, dtype=imgs.dtype), torch.arange(W, dtype=imgs.dtype), indexing="ij")
    P = torch.stack([xs, ys], -1).reshape(1, -1, 2)
    S = P @ A.transpose(1, 2) + t[:, None, :]
    gx = (2 * S[..., 0] + 1) / W - 1
    gy = (2 * S[..., 1] + 1) / H - 1
    grid = torch.stack([gx, gy], -1).reshape(B, H, W, 2)
    return F.grid_sample(imgs[:, None], grid, mode="bilinear", padding_mode="zeros", align_corners=False)[:, 0]


imgs = torch.rand(4, 32, 32)
print(warp_batch(imgs, torch.eye(2).repeat(4, 1, 1), torch.zeros(4, 2)).shape)
