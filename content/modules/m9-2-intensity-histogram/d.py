import math
import torch


def soft_histogram(x, bins=64, sigma=None):
    ...


class ToneCurve(torch.nn.Module):
    def __init__(self, n=64):
        super().__init__()
        ...

    def forward(self, x):
        ...


def learn_equalization(x, steps=300, lr=0.05):
    ...


from skimage import data
x = torch.tensor(data.moon(), dtype=torch.float32) / 255
curve = learn_equalization(x)
print(curve(torch.linspace(0, 1, 5)))
