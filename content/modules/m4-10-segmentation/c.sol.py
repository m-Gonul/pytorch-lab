import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def conv_matrix(w, H, s=1, p=0):
    k = w.shape[0]
    Ho = (H + 2 * p - k) // s + 1
    oi, oj, u, v = np.meshgrid(np.arange(Ho), np.arange(Ho), np.arange(k), np.arange(k), indexing="ij")
    r, c = oi * s + u - p, oj * s + v - p
    ok = (r >= 0) & (r < H) & (c >= 0) & (c < H)
    M = np.zeros((Ho * Ho, H * H))
    np.add.at(M, ((oi * Ho + oj)[ok], (r * H + c)[ok]), np.broadcast_to(w, oi.shape)[ok])
    return M


def tconv2d(x, w, s=1, p=0):
    Ci, H, W = x.shape
    k = w.shape[-1]
    Co = w.shape[1]
    contrib = np.einsum("chw,couv->ohwuv", x, w)
    R = (np.arange(H)[:, None, None, None] * s + np.arange(k)[None, None, :, None])
    Cc = (np.arange(W)[None, :, None, None] * s + np.arange(k)[None, None, None, :])
    out = np.zeros((Co, (H - 1) * s + k, (W - 1) * s + k))
    np.add.at(out, (slice(None), R, Cc), contrib)
    return out[:, p:out.shape[1] - p, p:out.shape[2] - p]


def tconv_zero_insert(x, w, s=1, p=0):
    Ci, H, W = x.shape
    k = w.shape[-1]
    z = np.zeros((Ci, (H - 1) * s + 1, (W - 1) * s + 1))
    z[:, ::s, ::s] = x
    q = k - 1 - p
    z = np.pad(z, ((0, 0), (q, q), (q, q)))
    wf = w[:, :, ::-1, ::-1].transpose(1, 0, 2, 3)
    return np.einsum("chwuv,ocuv->ohw", sliding_window_view(z, (k, k), axis=(1, 2)), wf)


def bilinear_kernel(f):
    k = 2 * f - f % 2
    c = f - 1 if k % 2 == 1 else f - 0.5
    t = 1 - np.abs(np.arange(k) - c) / f
    return np.outer(t, t)


def confusion(pred, gt, C):
    return np.bincount(gt.ravel() * C + pred.ravel(), minlength=C * C).reshape(C, C)


def miou(cm):
    tp = np.diag(cm).astype(float)
    iou = tp / (cm.sum(0) + cm.sum(1) - tp)
    return iou, float(np.nanmean(iou))
