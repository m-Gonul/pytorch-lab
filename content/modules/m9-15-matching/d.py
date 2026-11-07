import math
import numpy as np
import cv2
import torch


def preprocess(p):
    ...


def target(h, w, s=2.0):
    ...


class Mosse:
    def __init__(self, frame, cx, cy, w=64, h=64, lam=1e-2, n_aug=8, lr=0.125, seed=0):
        ...

    def update(self, frame):
        ...


f = np.random.default_rng(0).random((120, 160)).astype(np.float32) * 255
tr = Mosse(f, 80, 60)
print(tr.update(f))
