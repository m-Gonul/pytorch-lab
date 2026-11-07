import torch
import torch.nn.functional as F


def _sobel(img):
    k = torch.tensor([[[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]], [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=img.dtype)
    C = img.shape[1]
    w = k[:, None].repeat(C, 1, 1, 1)
    out = F.conv2d(F.pad(img, (1, 1, 1, 1), mode="replicate"), w, groups=C)
    return out[:, 0::2], out[:, 1::2]


def di_zenzo(img):
    gx, gy = _sobel(img)
    gxx, gyy, gxy = (gx * gx).sum(1), (gy * gy).sum(1), (gx * gy).sum(1)
    th = 0.5 * torch.atan2(2 * gxy, gxx - gyy)
    Fv = 0.5 * ((gxx + gyy) + (gxx - gyy) * torch.cos(2 * th) + 2 * gxy * torch.sin(2 * th))
    return Fv.clamp_min(0).sqrt(), th


def channel_sum_grad(img):
    gx, gy = _sobel(img)
    return torch.sqrt(gx * gx + gy * gy).sum(1)


def mahalanobis_mask(img, samples, D0):
    a = samples.mean(0)
    Ci = torch.linalg.inv(torch.cov(samples.T))
    z = img.permute(0, 2, 3, 1) - a
    d2 = torch.einsum("bhwi,ij,bhwj->bhw", z, Ci, z)
    return d2.clamp_min(0).sqrt() <= D0


def box_mask(img, samples, k=1.25):
    a, s = samples.mean(0), samples.std(0)
    z = img.permute(0, 2, 3, 1)
    return ((z - a).abs() <= k * s).all(-1)


x = torch.rand(1, 3, 16, 16)
print(di_zenzo(x)[0].shape)
