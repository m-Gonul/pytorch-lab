import torch
import torch.nn.functional as F


def filter2d(img, k, border="reflect101"):
    """img (H, W), k (m, n): korelasyon, çıkış aynı boyut."""
    ...


def recover_kernel(x, y, ksize, steps=500, lr=0.1):
    ...


from skimage import data
x = torch.tensor(data.camera(), dtype=torch.float32)[:128, :128] / 255
print(filter2d(x, torch.ones(3, 3) / 9).shape)
