import numpy as np
import cv2


def preprocess(prev, cur):
    m = np.maximum(prev, cur).astype(np.float64)
    y = m @ np.array([0.299, 0.587, 0.114])
    small = cv2.resize(y, (84, 110), interpolation=cv2.INTER_AREA)
    return np.clip(np.rint(small[18:102]), 0, 255).astype(np.uint8)


class FrameStack:
    def __init__(self, k=4):
        self.k = k
        self.buf = None

    def reset(self, f):
        self.buf = np.repeat(f[None], self.k, 0)

    def push(self, f):
        self.buf = np.concatenate([self.buf[1:], f[None]], 0)

    def state(self):
        return self.buf


def clip_reward(r):
    return float(np.sign(r))


rng = np.random.default_rng(0)
a, b = rng.integers(0, 256, (2, 210, 160, 3), dtype=np.uint8)
print(preprocess(a, b).shape)
