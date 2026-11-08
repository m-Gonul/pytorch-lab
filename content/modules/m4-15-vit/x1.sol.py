import numpy as np


def _lin_idx(g, ng):
    c = (np.arange(ng) + 0.5) * g / ng - 0.5
    c = np.clip(c, 0, g - 1)
    i0 = np.floor(c).astype(int)
    i1 = np.minimum(i0 + 1, g - 1)
    return i0, i1, c - i0


def interp_pos(pos, new_g):
    g = int(round(np.sqrt(len(pos))))
    G = pos.reshape(g, g, -1)
    r0, r1, wr = _lin_idx(g, new_g)
    c0, c1, wc = _lin_idx(g, new_g)
    top = G[r0][:, c0] * (1 - wc)[None, :, None] + G[r0][:, c1] * wc[None, :, None]
    bot = G[r1][:, c0] * (1 - wc)[None, :, None] + G[r1][:, c1] * wc[None, :, None]
    return (top * (1 - wr)[:, None, None] + bot * wr[:, None, None]).reshape(new_g * new_g, -1)


def _sincos_1d(pos, D):
    w = 10000.0 ** (-np.arange(D // 2) / (D // 2))
    a = pos[:, None] * w[None]
    return np.concatenate([np.sin(a), np.cos(a)], 1)


def sincos_2d(g, D):
    yy, xx = np.divmod(np.arange(g * g), g)
    return np.concatenate([_sincos_1d(yy.astype(float), D // 2), _sincos_1d(xx.astype(float), D // 2)], 1)
