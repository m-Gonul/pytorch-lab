import torch
import torch.nn.functional as F
from torch import nn


class Denoiser(nn.Module):
    def __init__(self, c=32, depth=4):
        super().__init__()
        ...

    def forward(self, x):
        ...


def train_denoiser(img, sigma, steps=500, batch=16, patch=48, seed=0):
    ...


from skimage import data
img = torch.tensor(data.camera(), dtype=torch.float32) / 255
m = train_denoiser(img, 25 / 255, steps=20)
print(m(img[None, None, :64, :64]).shape)
