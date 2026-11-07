import math
import numpy as np
import cv2


def _offsets(d, ss):
    r = max(d // 2 if d > 0 else round(1.5 * ss), 1)
    return r, [(i, j, np.float32(math.exp(-(i * i + j * j) / (2 * ss * ss)))) for i in range(-r, r + 1) for j in range(-r, r + 1) if math.sqrt(i * i + j * j) <= r]


def bilateral_color(img, d, sc, ss):
    r, offs = _offsets(d, ss)
    H, W, _ = img.shape
    P = cv2.copyMakeBorder(img, r, r, r, r, cv2.BORDER_REFLECT_101).astype(np.float32)
    c = img.astype(np.float32)
    num = np.zeros((H, W, 3), np.float32); den = np.zeros((H, W, 1), np.float32)
    for i, j, ws in offs:
        v = P[r + i:r + i + H, r + j:r + j + W]
        diff = np.abs(v - c).sum(-1, keepdims=True)
        w = ws * np.exp(-diff ** 2 / (2 * sc * sc)).astype(np.float32)
        num += w * v; den += w
    return np.clip(np.rint(num / den), 0, 255).astype(np.uint8)


def joint_bilateral(img, guide, d, sc, ss):
    r, offs = _offsets(d, ss)
    H, W = guide.shape
    P = cv2.copyMakeBorder(img, r, r, r, r, cv2.BORDER_REFLECT_101).astype(np.float64)
    G = cv2.copyMakeBorder(guide, r, r, r, r, cv2.BORDER_REFLECT_101).astype(np.float64)
    g0 = guide.astype(np.float64)
    num = np.zeros(img.shape); den = np.zeros(img.shape[:2])
    for i, j, ws in offs:
        w = float(ws) * np.exp(-(G[r + i:r + i + H, r + j:r + j + W] - g0) ** 2 / (2 * sc * sc))
        v = P[r + i:r + i + H, r + j:r + j + W]
        num += (w[..., None] if img.ndim == 3 else w) * v; den += w
    out = num / (den[..., None] if img.ndim == 3 else den)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


from skimage import data
print(bilateral_color(data.astronaut()[:64, :64], 7, 30, 5).shape)
