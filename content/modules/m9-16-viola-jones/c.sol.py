import numpy as np

SIGNS = {"type-2-x": (-1, 1), "type-2-y": (-1, 1), "type-3-x": (-1, 1, -1), "type-3-y": (-1, 1, -1), "type-4": (-1, 1, -1, 1)}


def rect_table(coords, kinds):
    K = len(coords)
    R = np.zeros((K, 4, 4), np.int64)
    S = np.zeros((K, 4))
    for j, (c, k) in enumerate(zip(coords, kinds)):
        for i, ((r0, c0), (r1, c1)) in enumerate(c):
            R[j, i] = [r0, c0, r1 + 1, c1 + 1]
            S[j, i] = SIGNS[k][i]
    return R, S


def haar_matrix(X, R, S):
    I = np.pad(X.astype(np.float64), ((0, 0), (1, 0), (1, 0))).cumsum(1).cumsum(2)
    r0, c0, r1, c1 = R[..., 0], R[..., 1], R[..., 2], R[..., 3]
    box = I[:, r1, c1] - I[:, r0, c1] - I[:, r1, c0] + I[:, r0, c0]
    return (box * S).sum(-1)


def best_stump(f, y, w):
    o = np.argsort(f, kind="stable")
    fs, ys, ws = f[o], y[o], w[o]
    Sp, Sn = np.cumsum(ws * ys), np.cumsum(ws * (1 - ys))
    Tp, Tn = Sp[-1], Sn[-1]
    last = np.r_[fs[1:] != fs[:-1], True]
    e1 = np.where(last, Sn + (Tp - Sp), np.inf)
    e2 = np.where(last, Sp + (Tn - Sn), np.inf)
    mid = np.r_[(fs[:-1] + fs[1:]) / 2, fs[-1] + 1]
    i1, i2 = np.argmin(e1), np.argmin(e2)
    if e1[i1] <= e2[i2]:
        return float(e1[i1]), float(mid[i1]), 1
    return float(e2[i2]), float(mid[i2]), -1


from skimage import data
from skimage.feature import haar_like_feature_coord
c, k = haar_like_feature_coord(25, 25, "type-2-x")
R, S = rect_table(c[:10], k[:10])
print(haar_matrix(data.lfw_subset()[:3], R, S).shape)
