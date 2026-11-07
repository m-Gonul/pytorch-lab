import numpy as np
import torch
import torch.nn.functional as F


def dilate_t(f, w):
    k = w.shape[0]
    r = k // 2
    U = F.unfold(F.pad(f, (r, r, r, r), value=-1e4), k)
    return (U + w.flip(0, 1).reshape(1, -1, 1)).max(1).values.view(f.shape)


def learn_se(img, target_se, steps=1500, lr=0.02, patch=128, seed=0):
    import cv2
    g = torch.Generator().manual_seed(seed)
    X = torch.tensor(img)[None, None]
    Y = torch.tensor(cv2.dilate(img, target_se))[None, None]
    H, W = img.shape
    w = torch.zeros(3, 3, requires_grad=True)
    opt = torch.optim.Adam([w], lr=lr)
    for _ in range(steps):
        i = int(torch.randint(0, H - patch, (1,), generator=g))
        j = int(torch.randint(0, W - patch, (1,), generator=g))
        loss = F.mse_loss(dilate_t(X[..., i:i + patch, j:j + patch], w), Y[..., i:i + patch, j:j + patch])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return w.detach()


def tophat_t(f, k):
    r = k // 2
    er = -F.max_pool2d(-f, k, 1, r)
    op = F.max_pool2d(er, k, 1, r)
    return f - op


f = torch.rand(1, 1, 8, 8)
print(dilate_t(f, torch.zeros(3, 3)).shape)
