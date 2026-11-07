import torch
import torch.nn.functional as F


def canny_torch(imgs, lo, hi, l2=False):
    """imgs (B, H, W) uint8 → (B, H, W) uint8 (0/255)."""
    ...


imgs = torch.randint(0, 256, (2, 32, 32), dtype=torch.uint8)
print(canny_torch(imgs, 50, 150).shape)
