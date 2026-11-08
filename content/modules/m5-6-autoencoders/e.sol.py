import time
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvAE(nn.Module):
    def __init__(self, code=16):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(1, 16, 3, 2, 1), nn.ReLU(), nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(), nn.Flatten(), nn.Linear(32 * 16, code))
        self.dec = nn.Sequential(nn.Linear(code, 32 * 16), nn.ReLU(), nn.Unflatten(1, (32, 4, 4)), nn.ConvTranspose2d(32, 16, 3, 2, 1, 1), nn.ReLU(),
                                 nn.ConvTranspose2d(16, 1, 3, 2, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return self.dec(self.enc(x))


class DenseAE(nn.Module):
    def __init__(self, code=16):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, code), nn.Linear(code, 128), nn.ReLU(), nn.Linear(128, 256), nn.Sigmoid(), nn.Unflatten(1, (1, 16, 16)))

    def forward(self, x):
        return self.net(x)
