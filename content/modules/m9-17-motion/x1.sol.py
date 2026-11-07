import numpy as np
import cv2


def temporal_mean(frames, k):
    F = np.asarray(frames, np.float64)
    T = len(F)
    c = np.cumsum(np.pad(F, ((1, 0), (0, 0), (0, 0))), 0)
    lo, hi = np.maximum(np.arange(T) - k, 0), np.minimum(np.arange(T) + k + 1, T)
    return (c[hi] - c[lo]) / (hi - lo)[:, None, None]


def gated_mean(frames, k, sigma, c=3.0):
    F = np.asarray(frames, np.float64)
    T = len(F)
    out = np.empty_like(F)
    for t in range(T):
        W = F[max(t - k, 0):t + k + 1]
        m = np.abs(W - F[t]) <= c * sigma
        out[t] = (W * m).sum(0) / m.sum(0)
    return out


def spatial(frames, s=1.5):
    return np.stack([cv2.GaussianBlur(np.asarray(f, np.float64), (0, 0), s) for f in frames])


print("zamansal")
