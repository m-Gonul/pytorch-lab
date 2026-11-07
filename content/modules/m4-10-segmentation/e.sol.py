import time
import torch
import torch.nn as nn


class _Resize(nn.Module):
    def __init__(self, C, mode):
        super().__init__()
        self.mode = mode
        self.conv = nn.Conv2d(C, C, 3, padding=1)

    def forward(self, x):
        return self.conv(nn.functional.interpolate(x, scale_factor=2, mode=self.mode))


def upsamplers(C):
    return {
        "tconv k3 s2": nn.ConvTranspose2d(C, C, 3, 2, 1, output_padding=1),
        "tconv k4 s2": nn.ConvTranspose2d(C, C, 4, 2, 1),
        "nearest + conv3": _Resize(C, "nearest"),
        "bilinear + conv3": _Resize(C, "bilinear"),
        "PixelShuffle": nn.Sequential(nn.Conv2d(C, 4 * C, 1), nn.PixelShuffle(2)),
    }


def structural_artifact(m):
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() > 1:
                p.fill_(1.0 / p[0].numel())
            else:
                p.zero_()
        C = m.in_channels if isinstance(m, nn.ConvTranspose2d) else [q for q in m.modules() if isinstance(q, nn.Conv2d)][0].in_channels
        y = m(torch.ones(1, C, 16, 16))[0, :, 4:-4, 4:-4]
        return float((y.max() - y.min()) / y.mean())
