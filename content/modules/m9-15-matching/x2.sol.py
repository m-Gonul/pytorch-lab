import numpy as np


def _A(src, dst):
    x, y = src[:, 0], src[:, 1]
    u, v = dst[:, 0], dst[:, 1]
    z, o = np.zeros_like(x), np.ones_like(x)
    return np.concatenate([np.stack([-x, -y, -o, z, z, z, u * x, u * y, u], 1), np.stack([z, z, z, -x, -y, -o, v * x, v * y, v], 1)])


def _norm(p):
    c = p.mean(0)
    s = np.sqrt(2) / np.mean(np.linalg.norm(p - c, axis=1))
    return (p - c) * s, np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])


def dlt_raw(src, dst):
    H = np.linalg.svd(_A(src, dst))[2][-1].reshape(3, 3)
    return H / H[2, 2]


def dlt_norm(src, dst):
    s, Ts = _norm(src)
    d, Td = _norm(dst)
    H = np.linalg.inv(Td) @ np.linalg.svd(_A(s, d))[2][-1].reshape(3, 3) @ Ts
    return H / H[2, 2]


def cond_A(src, dst, normalize):
    if normalize:
        src, dst = _norm(src)[0], _norm(dst)[0]
    return float(np.linalg.cond(_A(src, dst)))


s = np.random.default_rng(0).uniform(0, 1000, (8, 2))
print(cond_A(s, s + 3, False), cond_A(s, s + 3, True))
