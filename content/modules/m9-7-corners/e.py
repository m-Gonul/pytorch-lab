import torch
import torch.nn.functional as F


def refine_subpix(img, q, win=5, iters=10):
    """img (H, W) uint8 dizi, q (N, 2) [x, y]. (N, 2) tensör döndür."""
    ...


import numpy as np
sq = np.zeros((32, 32), np.uint8); sq[8:24, 8:24] = 200
print(refine_subpix(sq, np.array([[9.0, 7.0]])))  # köşe ≈ (7.5, 7.5)
