import torch
import torch.nn.functional as F


def vector_median(img, k):
    """img (H, W, 3) uint8 tensör."""
    ...


print(vector_median(torch.randint(0, 256, (8, 8, 3), dtype=torch.uint8), 3).shape)
