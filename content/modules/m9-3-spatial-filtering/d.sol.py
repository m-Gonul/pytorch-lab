import torch
import torch.nn.functional as F


def filter2d(img, k, border="reflect101"):
    """img (H, W), k (m, n): korelasyon, çıkış aynı boyut."""
    m, n = k.shape
    a, b = m // 2, n // 2
    mode = {"constant": "constant", "replicate": "replicate", "reflect101": "reflect"}[border]
    x = F.pad(img[None, None], (b, b, a, a), mode=mode)
    return F.conv2d(x, k[None, None].to(img.dtype))[0, 0]


def recover_kernel(x, y, ksize, steps=500, lr=0.1):
    w = torch.zeros(ksize, ksize, requires_grad=True)
    opt = torch.optim.Adam([w], lr=lr)
    for _ in range(steps):
        loss = ((filter2d(x, w) - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return w.detach()


from skimage import data
x = torch.tensor(data.camera(), dtype=torch.float32)[:128, :128] / 255
print(filter2d(x, torch.ones(3, 3) / 9).shape)
