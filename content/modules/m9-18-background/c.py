import numpy as np


def vibe_init(frame, N=20, seed=0):
    ...


def vibe_step(samples, frame, rng, R=20, nmin=2, phi=16):
    ...


def vibe_run(frames, N=20, R=20, nmin=2, phi=16, seed=0):
    ...


f = np.full((20, 30), 100, np.uint8)
print(vibe_run(np.stack([f, f, f + 80])).sum((1, 2)))
