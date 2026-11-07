import numpy as np


def connected_components(mask):
    H, W = mask.shape
    big = H * W + 1
    lab = np.where(mask, np.arange(H * W).reshape(H, W), big)
    while True:
        p = np.pad(lab, 1, constant_values=big)
        new = np.minimum.reduce([lab, p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]])
        new = np.where(mask, new, big)
        if np.array_equal(new, lab):
            break
        lab = new
    u, inv = np.unique(lab[mask], return_inverse=True)
    out = np.zeros((H, W), int)
    out[mask] = inv + 1
    return out, len(u)


def instances_from_semantic(sem, C):
    res = []
    for c in range(1, C):
        lab, K = connected_components(sem == c)
        res += [(c, lab == k) for k in range(1, K + 1)]
    return res
