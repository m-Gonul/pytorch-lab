import numpy as np
import torch
from skimage import data
from skimage.feature import haar_like_feature_coord

# --- d'den hazır: öznitelik tablosu, normalize öznitelikler, AdaBoost ---
SIGNS = {"type-2-x": (-1, 1), "type-2-y": (-1, 1), "type-3-x": (-1, 1, -1), "type-3-y": (-1, 1, -1), "type-4": (-1, 1, -1, 1)}


def feature_table(K=3000, seed=0, size=25):
    rng = np.random.default_rng(seed)
    coords, kinds = [], []
    for k in SIGNS:
        c, t = haar_like_feature_coord(size, size, k)
        coords += list(c); kinds += list(t)
    sel = rng.choice(len(coords), K, replace=False)
    R = np.zeros((K, 4, 4), np.int64); S = np.zeros((K, 4))
    for j, i in enumerate(sel):
        for q, ((r0, c0), (r1, c1)) in enumerate(coords[i]):
            R[j, q] = [r0, c0, r1 + 1, c1 + 1]; S[j, q] = SIGNS[kinds[i]][q]
    A = (S * (R[..., 2] - R[..., 0]) * (R[..., 3] - R[..., 1])).sum(1)
    return R, S, A


def normalized_features(X, R, S, A):
    I = np.pad(X.astype(np.float64), ((0, 0), (1, 0), (1, 0))).cumsum(1).cumsum(2)
    r0, c0, r1, c1 = R[..., 0], R[..., 1], R[..., 2], R[..., 3]
    F = ((I[:, r1, c1] - I[:, r0, c1] - I[:, r1, c0] + I[:, r0, c0]) * S).sum(-1)
    mu = X.reshape(len(X), -1).mean(1, keepdims=True); sd = X.reshape(len(X), -1).std(1, keepdims=True) + 1e-8
    return (F - mu * A) / sd


def train(F, y, T):
    y = y.double()
    w = torch.where(y == 1, 0.5 / y.sum(), 0.5 / (1 - y).sum())
    H = []
    for _ in range(T):
        w = w / w.sum()
        o = F.argsort(0); Fs = torch.gather(F, 0, o); ys, ws = y[o], w[o]
        Sp, Sn = torch.cumsum(ws * ys, 0), torch.cumsum(ws * (1 - ys), 0)
        e1, e2 = Sn + ((w * y).sum() - Sp), Sp + ((w * (1 - y)).sum() - Sn)
        last = torch.cat([Fs[1:] != Fs[:-1], torch.ones(1, F.shape[1], dtype=torch.bool)])
        e1, e2 = torch.where(last, e1, torch.inf), torch.where(last, e2, torch.inf)
        mid = torch.cat([(Fs[:-1] + Fs[1:]) / 2, Fs[-1:] + 1])
        m1, i1 = e1.min(0); m2, i2 = e2.min(0)
        err = torch.minimum(m1, m2); j = int(err.argmin())
        pol = 1 if m1[j] <= m2[j] else -1
        thr = float(mid[i1[j] if pol == 1 else i2[j], j])
        beta = max(float(err[j]), 1e-10) / (1 - float(err[j]))
        h = ((pol * F[:, j]) <= (pol * thr)).double()
        w = w * torch.where(h == y, beta, 1.0)
        H.append((j, thr, pol, float(np.log(1 / beta))))
    return H


# --- senin kodun ---
def make_cascade(F_pos, H, sizes=(2, 10, 30)):
    ...


def scan(img, R, S, A, H, cascade, win=25):
    ...


def nms(boxes, r=12):
    ...


print("kaskad")
