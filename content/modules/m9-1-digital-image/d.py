import math
import torch
import torch.nn.functional as F


def warp_affine(img, A, t):
    """img (H, W), A (2, 2), t (2,): kaynak = A @ [x, y] + t (piksel birimi)."""
    ...


def register(fixed, moving, steps=300, lr=0.05):
    ...


from skimage import data
img = torch.tensor(data.camera(), dtype=torch.float32)[::4, ::4] / 255
print(warp_affine(img, torch.eye(2), torch.tensor([3.0, 0.0])).shape)
