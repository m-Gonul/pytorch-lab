import math
import numpy as np
import torch


def _init_centers(L, S):
    H, W, _ = L.shape
    P = np.pad(L, ((1, 1), (1, 1), (0, 0)), mode="edge")
    G = ((P[1:-1, 2:] - P[1:-1, :-2]) ** 2).sum(-1) + ((P[2:, 1:-1] - P[:-2, 1:-1]) ** 2).sum(-1)
    cs = []
    for y in range(S // 2, H, S):
        for x in range(S // 2, W, S):
            ys, xs = slice(max(y - 1, 0), min(y + 2, H)), slice(max(x - 1, 0), min(x + 2, W))
            i = np.argmin(G[ys, xs])
            yy, xx = ys.start + i // (xs.stop - xs.start), xs.start + i % (xs.stop - xs.start)
            cs.append([*L[yy, xx], yy, xx])
    return torch.tensor(np.array(cs, np.float64))


def slic(lab, K, c=10.0, iters=10):
    H, W, _ = lab.shape
    S = int(round(math.sqrt(H * W / K)))
    C = _init_centers(lab, S)
    Lt = torch.tensor(lab, dtype=torch.float64).reshape(-1, 3)
    yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float64), torch.arange(W, dtype=torch.float64), indexing="ij")
    pos = torch.stack([yy, xx], -1).reshape(-1, 2)
    F = torch.cat([Lt, pos], 1)
    lab_ = torch.full((H * W,), -1, dtype=torch.int64)
    inf = torch.tensor(float("inf"), dtype=torch.float64)
    for _ in range(iters):
        dy = (pos[:, None, 0] - C[None, :, 3]).abs()
        dx = (pos[:, None, 1] - C[None, :, 4]).abs()
        ok = (dy <= S) & (dx <= S)
        D = torch.sqrt(torch.cdist(Lt, C[:, :3]) ** 2 / c ** 2 + (dy ** 2 + dx ** 2) / S ** 2)
        D = torch.where(ok, D, inf)
        lab_ = torch.where(ok.any(1), D.argmin(1), lab_)
        valid = (lab_ >= 0).double()
        idx = lab_.clamp_min(0)
        cnt = torch.zeros(len(C), dtype=torch.float64).index_add_(0, idx, valid)
        sm = torch.zeros_like(C).index_add_(0, idx, F * valid[:, None])
        C = torch.where(cnt[:, None] > 0, sm / cnt.clamp_min(1)[:, None], C)
    return lab_.view(H, W).numpy()


from skimage import data, color
L = color.rgb2lab(data.astronaut()[:64, :64])
print(np.unique(slic(L, 16)).size)
