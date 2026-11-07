import torch
import torch.nn.functional as F


def adaptive_median_vec(img, smax=7):
    """img (H, W) uint8 tensör. Piksel döngüsü yok."""
    ...


img = torch.randint(0, 256, (16, 16), dtype=torch.uint8)
print(adaptive_median_vec(img).shape)
