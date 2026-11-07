import torch
import torch.nn.functional as F


def ms_ssim(x, y, R=255.0, weights=(0.0448, 0.2856, 0.3001, 0.2363, 0.1333)):
    ...


x = torch.rand(2, 192, 192, dtype=torch.float64) * 255
print(ms_ssim(x, x))
