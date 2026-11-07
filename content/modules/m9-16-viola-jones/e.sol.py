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
def _score(F, H):
    s = torch.zeros(F.shape[0], dtype=torch.float64)
    for j, t, p, a in H:
        s = s + a * ((p * F[:, j]) <= (p * t)).double()
    return s


def make_cascade(F_pos, H, sizes=(2, 10, 30)):
    out = []
    for k in sizes:
        s = _score(F_pos, H[:k])
        half = sum(a for *_, a in H[:k]) / 2
        out.append((k, min(float(s.min()), half)))
    return out


def scan(img, R, S, A, H, cascade, win=25):
    f = torch.as_tensor(np.asarray(img, np.float64))
    I = torch.nn.functional.pad(f, (1, 0, 1, 0)).cumsum(0).cumsum(1)
    I2 = torch.nn.functional.pad(f * f, (1, 0, 1, 0)).cumsum(0).cumsum(1)
    Hh, Ww = f.shape
    ys, xs = torch.meshgrid(torch.arange(Hh - win + 1), torch.arange(Ww - win + 1), indexing="ij")
    ys, xs = ys.flatten(), xs.flatten()
    n = win * win
    box = lambda J, y, x: J[y + win, x + win] - J[y, x + win] - J[y + win, x] + J[y, x]
    mu = box(I, ys, xs) / n
    sd = (box(I2, ys, xs) / n - mu * mu).clamp_min(0).sqrt() + 1e-8
    Rt, St, At = torch.as_tensor(R), torch.as_tensor(S), torch.as_tensor(A)
    used = sorted({j for j, *_ in H})
    col = {j: i for i, j in enumerate(used)}
    Ru, Su, Au = Rt[used], St[used], At[used]
    alive = torch.arange(len(ys))
    counts = []
    s = None
    for k, thr in cascade:
        counts.append(len(alive))
        y, x = ys[alive][:, None, None], xs[alive][:, None, None]
        r0, c0, r1, c1 = Ru[..., 0], Ru[..., 1], Ru[..., 2], Ru[..., 3]
        v = ((I[y + r1, x + c1] - I[y + r0, x + c1] - I[y + r1, x + c0] + I[y + r0, x + c0]) * Su).sum(-1)
        Fw = (v - mu[alive][:, None] * Au) / sd[alive][:, None]
        s = torch.zeros(len(alive), dtype=torch.float64)
        for j, t, p, a in H[:k]:
            s = s + a * ((p * Fw[:, col[j]]) <= (p * t)).double()
        keep = s >= thr
        alive, s = alive[keep], s[keep]
    return [(int(xs[i]), int(ys[i]), float(v)) for i, v in zip(alive, s)], counts


def nms(boxes, r=12):
    out = []
    for x, y, s in sorted(boxes, key=lambda b: -b[2]):
        if all((x - a) ** 2 + (y - b) ** 2 >= r * r for a, b, _ in out):
            out.append((x, y, s))
    return out


print("kaskad")
