import math
import numpy as np
import cv2
import torch


def preprocess(p):
    p = np.log(p.astype(np.float64) + 1)
    p = (p - p.mean()) / (p.std() + 1e-5)
    h, w = p.shape
    return torch.tensor(p * np.outer(np.hanning(h), np.hanning(w)))


def target(h, w, s=2.0):
    y, x = np.mgrid[:h, :w]
    g = np.exp(-((x - w // 2) ** 2 + (y - h // 2) ** 2) / (2 * s * s))
    return torch.fft.fft2(torch.tensor(g))


class Mosse:
    def __init__(self, frame, cx, cy, w=64, h=64, lam=1e-2, n_aug=8, lr=0.125, seed=0):
        self.w, self.h, self.lam, self.lr = w, h, lam, lr
        self.cx, self.cy = float(cx), float(cy)
        self.G = target(h, w)
        rng = np.random.default_rng(seed)
        p = cv2.getRectSubPix(frame, (w, h), (self.cx, self.cy))
        self.A = torch.zeros(h, w, dtype=torch.complex128)
        self.B = torch.zeros(h, w, dtype=torch.complex128)
        for i in range(n_aug):
            q = p
            if i:
                M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-10, 10), rng.uniform(0.95, 1.05))
                q = cv2.warpAffine(p, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            F = torch.fft.fft2(preprocess(q))
            self.A += self.G * F.conj()
            self.B += F * F.conj()

    def update(self, frame):
        F = torch.fft.fft2(preprocess(cv2.getRectSubPix(frame, (self.w, self.h), (self.cx, self.cy))))
        r = torch.fft.ifft2(F * self.A / (self.B + self.lam)).real
        i = int(r.argmax())
        y, x = i // self.w, i % self.w
        side = torch.ones_like(r, dtype=torch.bool)
        side[max(y - 5, 0):y + 6, max(x - 5, 0):x + 6] = False
        psr = float((r[y, x] - r[side].mean()) / r[side].std())
        self.cx += x - self.w // 2
        self.cy += y - self.h // 2
        F = torch.fft.fft2(preprocess(cv2.getRectSubPix(frame, (self.w, self.h), (self.cx, self.cy))))
        self.A = self.lr * self.G * F.conj() + (1 - self.lr) * self.A
        self.B = self.lr * F * F.conj() + (1 - self.lr) * self.B
        return self.cx, self.cy, psr


f = np.random.default_rng(0).random((120, 160)).astype(np.float32) * 255
tr = Mosse(f, 80, 60)
print(tr.update(f))
