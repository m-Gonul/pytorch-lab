import numpy as np


def encoder(z, Wq, Wk, Wv):
    s = (z @ Wq) @ (z @ Wk).T / np.sqrt(z.shape[1])
    a = np.exp(s - s.max(1, keepdims=True))
    return z + (a / a.sum(1, keepdims=True)) @ (z @ Wv)


def vit_forward(patches, E, cls, pos, W):
    z = np.concatenate([cls[None], patches @ E], 0)
    if pos is not None:
        z = z + pos
    return encoder(z, *W)
