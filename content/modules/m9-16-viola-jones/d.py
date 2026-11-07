import numpy as np
import torch
from skimage import data
from skimage.feature import haar_like_feature_coord

SIGNS = {"type-2-x": (-1, 1), "type-2-y": (-1, 1), "type-3-x": (-1, 1, -1), "type-3-y": (-1, 1, -1), "type-4": (-1, 1, -1, 1)}


def feature_table(K=5000, seed=0, size=25):
    """Rastgele K öznitelik: R (K,4,4) [r0,c0,r1+1,c1+1], S (K,4) işaret, A (K,) Σ işaret·alan."""
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
    """Pencere başına (f − μ·A)/σ."""
    I = np.pad(X.astype(np.float64), ((0, 0), (1, 0), (1, 0))).cumsum(1).cumsum(2)
    r0, c0, r1, c1 = R[..., 0], R[..., 1], R[..., 2], R[..., 3]
    F = ((I[:, r1, c1] - I[:, r0, c1] - I[:, r1, c0] + I[:, r0, c0]) * S).sum(-1)
    mu = X.reshape(len(X), -1).mean(1, keepdims=True); sd = X.reshape(len(X), -1).std(1, keepdims=True) + 1e-8
    return (F - mu * A) / sd


def best_stumps(F, y, w):
    ...


def adaboost(F, y, T):
    ...


def strong_score(F, H, k=None):
    ...


X = data.lfw_subset()
R, S, A = feature_table(500)
F = torch.tensor(normalized_features(X, R, S, A))
y = torch.tensor(np.r_[np.ones(100), np.zeros(100)])
print(adaboost(F, y, 3))
