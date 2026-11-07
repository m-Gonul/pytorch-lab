import torch
import torch.nn.functional as F
from torch import nn


class Denoiser(nn.Module):
    def __init__(self, c=32, depth=4):
        super().__init__()
        layers = [nn.Conv2d(1, c, 3, padding=1), nn.ReLU()]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(c, c, 3, padding=1), nn.ReLU()]
        layers += [nn.Conv2d(c, 1, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return x - self.net(x)


def train_denoiser(img, sigma, steps=500, batch=16, patch=48, seed=0):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    H, W = img.shape
    m = Denoiser()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for _ in range(steps):
        i = torch.randint(0, H - patch, (batch,), generator=g)
        j = torch.randint(0, W - patch, (batch,), generator=g)
        clean = torch.stack([img[a:a + patch, b:b + patch] for a, b in zip(i.tolist(), j.tolist())])[:, None]
        noisy = clean + sigma * torch.randn(clean.shape, generator=g)
        loss = F.mse_loss(m(noisy), clean)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return m


from skimage import data
img = torch.tensor(data.camera(), dtype=torch.float32) / 255
m = train_denoiser(img, 25 / 255, steps=20)
print(m(img[None, None, :64, :64]).shape)
