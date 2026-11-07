import numpy as np
import cv2
import torch
import torch.nn.functional as F


def _A(src, dst):
    x, y = src[..., 0], src[..., 1]
    u, v = dst[..., 0], dst[..., 1]
    z, o = torch.zeros_like(x), torch.ones_like(x)
    r1 = torch.stack([-x, -y, -o, z, z, z, u * x, u * y, u], -1)
    r2 = torch.stack([z, z, z, -x, -y, -o, v * x, v * y, v], -1)
    return torch.cat([r1, r2], -2)


def _norm(p):
    c = p.mean(0)
    s = 2 ** 0.5 / (p - c).norm(dim=1).mean()
    T = torch.tensor([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]], dtype=p.dtype)
    return (p - c) * s, T


def _dlt(src, dst):
    s, Ts = _norm(src)
    d, Td = _norm(dst)
    Hn = torch.linalg.svd(_A(s, d))[2][-1].reshape(3, 3)
    H = torch.linalg.inv(Td) @ Hn @ Ts
    return H / H[2, 2]


def _proj(H, p):
    q = torch.cat([p, torch.ones(len(p), 1, dtype=p.dtype)], 1) @ H.transpose(-1, -2)
    return q[..., :2] / q[..., 2:]


def ransac_t(src, dst, thr=3.0, iters=1000, seed=0):
    src, dst = torch.as_tensor(src, dtype=torch.float64), torch.as_tensor(dst, dtype=torch.float64)
    g = torch.Generator().manual_seed(seed)
    idx = torch.rand(iters, len(src), generator=g).argsort(1)[:, :4]
    Hs = torch.linalg.svd(_A(src[idx], dst[idx]))[2][:, -1].reshape(-1, 3, 3)
    err = (_proj(Hs, src) - dst).norm(dim=-1)
    best = int((err < thr).sum(1).argmax())
    m = err[best] < thr
    H = _dlt(src[m], dst[m])
    return H, (_proj(H, src) - dst).norm(dim=1) < thr


def stitch(img1, img2):
    sift = cv2.SIFT_create()
    k1, d1 = sift.detectAndCompute(img1, None)
    k2, d2 = sift.detectAndCompute(img2, None)
    knn = cv2.BFMatcher().knnMatch(d2, d1, k=2)
    good = [m for m, n in knn if m.distance < 0.75 * n.distance]
    p2 = np.array([k2[m.queryIdx].pt for m in good]); p1 = np.array([k1[m.trainIdx].pt for m in good])
    H, _ = ransac_t(p2, p1)
    h1, w1 = img1.shape[:2]; h2, w2 = img2.shape[:2]
    c2 = _proj(H, torch.tensor([[0, 0], [w2 - 1, 0], [w2 - 1, h2 - 1], [0, h2 - 1]], dtype=torch.float64))
    allc = torch.cat([c2, torch.tensor([[0, 0], [w1 - 1, h1 - 1]], dtype=torch.float64)])
    x0, y0 = allc.min(0).values.floor()
    x1, y1 = allc.max(0).values.ceil()
    W, Hh = int(x1 - x0) + 1, int(y1 - y0) + 1
    ys, xs = torch.meshgrid(torch.arange(Hh, dtype=torch.float64) + y0, torch.arange(W, dtype=torch.float64) + x0, indexing="ij")
    P = torch.stack([xs.flatten(), ys.flatten()], 1)

    def sample(img, pts):
        h, w = img.shape[:2]
        t = torch.tensor(img, dtype=torch.float64).permute(2, 0, 1)[None]
        dist = torch.tensor(cv2.distanceTransform(np.pad(np.ones((h, w), np.uint8), 1), cv2.DIST_L2, 3)[1:-1, 1:-1], dtype=torch.float64)[None, None]
        g = torch.stack([pts[:, 0] / (w - 1) * 2 - 1, pts[:, 1] / (h - 1) * 2 - 1], -1).view(1, Hh, W, 2)
        v = F.grid_sample(t, g, align_corners=True)[0]
        wgt = F.grid_sample(dist, g, align_corners=True)[0, 0]
        inside = (pts[:, 0] >= 0) & (pts[:, 0] <= w - 1) & (pts[:, 1] >= 0) & (pts[:, 1] <= h - 1)
        return v, wgt * inside.view(Hh, W)

    v1, w1_ = sample(img1, P)
    v2, w2_ = sample(img2, _proj(torch.linalg.inv(H), P))
    den = (w1_ + w2_).clamp_min(1e-9)
    pano = (v1 * w1_ + v2 * w2_) / den
    pano = pano.permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8).numpy()
    return pano, H


print("panorama")
