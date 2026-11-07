import torch
import torch.nn.functional as F


def vector_median(img, k):
    """img (H, W, 3) uint8 tensör."""
    H, W, _ = img.shape
    r = k // 2
    x = F.pad(img.permute(2, 0, 1)[None].float(), (r, r, r, r), mode="replicate")
    U = F.unfold(x, k)[0].view(3, k * k, H * W).permute(2, 1, 0)
    out = torch.empty(H * W, 3)
    step = 16384
    for s in range(0, H * W, step):
        u = U[s:s + step]
        idx = torch.cdist(u, u).sum(-1).argmin(-1)
        out[s:s + step] = u[torch.arange(len(u)), idx]
    return out.view(H, W, 3).to(torch.uint8)


print(vector_median(torch.randint(0, 256, (8, 8, 3), dtype=torch.uint8), 3).shape)
