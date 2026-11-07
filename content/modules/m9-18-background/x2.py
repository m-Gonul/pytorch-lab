import numpy as np

OFF = np.array([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)])


def vibe_init(frame, N=20, seed=0):
    rng = np.random.default_rng(seed)
    H, W = frame.shape
    P = np.pad(frame.astype(np.int16), 1, mode="edge")
    yy, xx = np.mgrid[:H, :W]
    dy, dx = rng.integers(-1, 2, (2, N, H, W))
    return P[yy + 1 + dy, xx + 1 + dx]


def vibe_step(samples, frame, rng, R=20, nmin=2, phi=16):
    N, H, W = samples.shape
    f = frame.astype(np.int16)
    fg = (np.abs(samples - f) < R).sum(0) < nmin
    bg = ~fg
    y, x = np.nonzero(bg & (rng.random((H, W)) < 1 / phi))
    samples[rng.integers(0, N, len(y)), y, x] = f[y, x]
    y, x = np.nonzero(bg & (rng.random((H, W)) < 1 / phi))
    o = OFF[rng.integers(0, 8, len(y))]
    ny, nx = np.clip(y + o[:, 0], 0, H - 1), np.clip(x + o[:, 1], 0, W - 1)
    samples[rng.integers(0, N, len(y)), ny, nx] = f[y, x]
    return fg


def ghost_curve(frames, ghost_mask, phi, seed=0):
    ...


def half_life(curve):
    ...


print("hayalet")
