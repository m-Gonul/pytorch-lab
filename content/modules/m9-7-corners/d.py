import math
import torch
import torch.nn.functional as F


def similarity(img, angle, scale):
    """Merkez etrafında döndür + ölçekle. (img2, A) döndürür; A (2, 3): p2 = A @ [x, y, 1]."""
    H, W = img.shape
    c = torch.tensor([W / 2, H / 2])
    th = math.radians(angle)
    Rm = torch.tensor([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]]) * scale
    inv = torch.linalg.inv(Rm)
    ys, xs = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij")
    src = (torch.stack([xs, ys], -1) - c) @ inv.T + c
    grid = torch.stack([src[..., 0] / (W - 1) * 2 - 1, src[..., 1] / (H - 1) * 2 - 1], -1)[None]
    img2 = F.grid_sample(img[None, None], grid, align_corners=True)[0, 0]
    return img2, torch.cat([Rm, (c - Rm @ c)[:, None]], 1)


def harris_torch(img, sigma_d=1.0, sigma_i=2.0, k=0.04):
    ...


def detect(R, n=200, r=4, border=16):
    ...


def repeatability(img, angle, scale, adapt=False, tol=2.0):
    ...


from skimage import data
img = torch.tensor(data.camera(), dtype=torch.float32) / 255
print(repeatability(img, 30, 1.0))
