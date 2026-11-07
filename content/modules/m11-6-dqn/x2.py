import numpy as np
import cv2


def preprocess(prev, cur):
    ...


class FrameStack:
    def __init__(self, k=4):
        ...

    def reset(self, f):
        ...

    def push(self, f):
        ...

    def state(self):
        ...


def clip_reward(r):
    ...


rng = np.random.default_rng(0)
a, b = rng.integers(0, 256, (2, 210, 160, 3), dtype=np.uint8)
print(preprocess(a, b).shape)
