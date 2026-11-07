import math
import torch
import torch.nn.functional as F


def bilateral_t(img, r, sc, ss):
    k = 2 * r + 1
    U = F.unfold(F.pad(img[None, None], (r, r, r, r), mode="reflect"), k)[0].view(k * k, *img.shape)
    i, j = torch.meshgrid(torch.arange(-r, r + 1.0), torch.arange(-r, r + 1.0), indexing="ij")
    d2 = (i ** 2 + j ** 2).flatten().to(img.dtype)
    sc, ss = torch.as_tensor(sc, dtype=img.dtype), torch.as_tensor(ss, dtype=img.dtype)
    ws = torch.exp(-d2 / (2 * ss ** 2)) * (d2 <= r * r)
    w = ws[:, None, None] * torch.exp(-(U - img) ** 2 / (2 * sc ** 2))
    return (w * U).sum(0) / w.sum(0)


def fit_bilateral(noisy, clean, r=3, steps=150, lr=0.05):
    p = torch.tensor([math.log(10.0), math.log(1.0)], requires_grad=True)
    opt = torch.optim.Adam([p], lr=lr)
    for _ in range(steps):
        loss = ((bilateral_t(noisy, r, p[0].exp(), p[1].exp()) - clean) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    sc, ss = p.detach().exp().tolist()
    return sc, ss


x = torch.rand(16, 16) * 255
print(bilateral_t(x, 2, 30.0, 2.0).shape)
