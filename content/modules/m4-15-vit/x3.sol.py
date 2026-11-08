import numpy as np


def window_partition(x, M):
    H, W, C = x.shape
    return x.reshape(H // M, M, W // M, M, C).transpose(0, 2, 1, 3, 4).reshape(-1, M * M, C)


def window_reverse(w, M, H, W):
    C = w.shape[-1]
    return w.reshape(H // M, W // M, M, M, C).transpose(0, 2, 1, 3, 4).reshape(H, W, C)


def shifted_window_mask(H, W, M, s):
    rows = np.select([np.arange(H) < H - M, np.arange(H) < H - s], [0, 1], 2)
    cols = np.select([np.arange(W) < W - M, np.arange(W) < W - s], [0, 1], 2)
    region = (rows[:, None] * 3 + cols[None])[..., None].astype(float)
    ids = window_partition(region, M)[..., 0]
    return ids[:, :, None] == ids[:, None, :]


def attn_cost(H, W, C, M):
    hw = H * W
    return 4 * hw * C * C + 2 * hw * hw * C, 4 * hw * C * C + 2 * M * M * hw * C
