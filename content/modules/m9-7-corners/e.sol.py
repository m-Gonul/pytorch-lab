import torch
import torch.nn.functional as F


def refine_subpix(img, q, win=5, iters=10):
    """img (H, W) uint8 dizi, q (N, 2) [x, y]. (N, 2) tensör döndür."""
    f = torch.as_tensor(img, dtype=torch.float32)[None, None]
    H, W = f.shape[-2:]
    q = torch.as_tensor(q, dtype=torch.float32).clone()
    r = torch.arange(-win - 1, win + 2, dtype=torch.float32)
    dy, dx = torch.meshgrid(r, r, indexing="ij")
    w = torch.exp(-(dx[1:-1, 1:-1] ** 2 + dy[1:-1, 1:-1] ** 2) / (2 * (win / 2) ** 2))
    for _ in range(iters):
        px = q[:, 0, None, None] + dx
        py = q[:, 1, None, None] + dy
        grid = torch.stack([px / (W - 1) * 2 - 1, py / (H - 1) * 2 - 1], -1)
        p = F.grid_sample(f.expand(len(q), 1, H, W), grid, align_corners=True)[:, 0]
        gx = (p[:, 1:-1, 2:] - p[:, 1:-1, :-2]) / 2
        gy = (p[:, 2:, 1:-1] - p[:, :-2, 1:-1]) / 2
        X, Y = px[:, 1:-1, 1:-1], py[:, 1:-1, 1:-1]
        a, b, c = (w * gx * gx).sum((1, 2)), (w * gx * gy).sum((1, 2)), (w * gy * gy).sum((1, 2))
        bx = (w * (gx * gx * X + gx * gy * Y)).sum((1, 2))
        by = (w * (gx * gy * X + gy * gy * Y)).sum((1, 2))
        A = torch.stack([torch.stack([a, b], -1), torch.stack([b, c], -1)], -2)
        q = torch.linalg.solve(A, torch.stack([bx, by], -1))
    return q


import numpy as np
sq = np.zeros((32, 32), np.uint8); sq[8:24, 8:24] = 200
print(refine_subpix(sq, np.array([[9.0, 7.0]])))  # köşe ≈ (7.5, 7.5)
