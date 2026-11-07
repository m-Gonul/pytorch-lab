import math
import torch
import torch.nn.functional as F


def warp_affine(img, A, t):
    """img (H, W), A (2, 2), t (2,): kaynak = A @ [x, y] + t (piksel birimi)."""
    H, W = img.shape
    ys, xs = torch.meshgrid(torch.arange(H, dtype=img.dtype), torch.arange(W, dtype=img.dtype), indexing="ij")
    P = torch.stack([xs, ys], -1)
    S = P @ A.T + t
    gx = (2 * S[..., 0] + 1) / W - 1
    gy = (2 * S[..., 1] + 1) / H - 1
    grid = torch.stack([gx, gy], -1)[None]
    return F.grid_sample(img[None, None], grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0, 0]


def _rot_about_center(theta, tx, ty, H, W):
    c, s = torch.cos(theta), torch.sin(theta)
    A = torch.stack([torch.stack([c, -s]), torch.stack([s, c])])
    ctr = torch.tensor([(W - 1) / 2, (H - 1) / 2])
    t = ctr - A @ ctr + torch.stack([tx, ty])
    return A, t


def register(fixed, moving, steps=300, lr=0.05):
    H, W = fixed.shape
    p = torch.zeros(3, requires_grad=True)
    opt = torch.optim.Adam([p], lr=lr)
    for _ in range(steps):
        A, t = _rot_about_center(p[0], p[1] * 10, p[2] * 10, H, W)
        loss = ((warp_affine(moving, A, t) - fixed)[16:-16, 16:-16] ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return p[0].item(), p[1].item() * 10, p[2].item() * 10


from skimage import data
img = torch.tensor(data.camera(), dtype=torch.float32)[::4, ::4] / 255
print(warp_affine(img, torch.eye(2), torch.tensor([3.0, 0.0])).shape)
