import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def scan_blocks(a, p):
    b = np.asarray(a).reshape(p, -1)
    loc = np.cumsum(b, 1)
    off = np.concatenate([[0], np.cumsum(loc[:, -1])[:-1]])
    return (loc + off[:, None]).ravel()


def integral_image(img):
    return np.cumsum(np.cumsum(np.asarray(img, np.int64), 0), 1)


def _to_u8(acc):
    return np.clip((acc + np.float32(0.5)).astype(np.int64), 0, 255).astype(np.uint8)


def conv2d(img, k):
    k = np.asarray(k, np.float32)
    r = k.shape[0] // 2
    f = np.pad(np.asarray(img, np.float32), r)
    win = sliding_window_view(f, k.shape)
    return _to_u8(np.einsum("yxuv,uv->yx", win, k))


def conv2d_separable(img, kx, ky):
    kx, ky = np.asarray(kx, np.float32), np.asarray(ky, np.float32)
    r = len(kx) // 2
    f = np.pad(np.asarray(img, np.float32), r)
    h = np.einsum("yxv,v->yx", sliding_window_view(f, len(kx), axis=1), kx)
    return _to_u8(np.einsum("yxu,u->yx", sliding_window_view(h, len(ky), axis=0), ky))


def matmul_tiled(A, B, T):
    n = A.shape[0]
    At = A.reshape(n // T, T, n // T, T)
    Bt = B.reshape(n // T, T, n // T, T)
    return np.einsum("ikjl,jlmn->ikmn", At, Bt).reshape(n, n)


print(scan_blocks(np.arange(1, 9), 4))
