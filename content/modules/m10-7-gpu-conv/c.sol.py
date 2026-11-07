import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def im2col(img, K):
    r = K // 2
    f = np.pad(np.asarray(img, np.float32), r)
    H, W = np.asarray(img).shape
    return sliding_window_view(f, (K, K)).reshape(H * W, K * K)


def conv_im2col(img, k):
    H, W = np.asarray(img).shape
    return (im2col(img, k.shape[0]) @ np.asarray(k, np.float32).ravel()).reshape(H, W)


def conv_multi(x, w):
    Cin, H, W = x.shape
    Cout, _, K, _ = w.shape
    r = K // 2
    f = np.pad(np.asarray(x, np.float32), ((0, 0), (r, r), (r, r)))
    cols = sliding_window_view(f, (K, K), axis=(1, 2))
    cols = cols.transpose(0, 3, 4, 1, 2).reshape(Cin * K * K, H * W)
    return (w.reshape(Cout, -1).astype(np.float32) @ cols).reshape(Cout, H, W)


def conv_fft(img, k):
    img = np.asarray(img, np.float64)
    H, W = img.shape
    K = k.shape[0]
    s = (H + K - 1, W + K - 1)
    F = np.fft.rfft2(img, s) * np.fft.rfft2(np.asarray(k, np.float64)[::-1, ::-1], s)
    full = np.fft.irfft2(F, s)
    r = K // 2
    return full[r:r + H, r:r + W]


def tile_intensity(BM, BN, BK):
    return 2 * BM * BN * BK / (4 * (BM * BK + BK * BN))


print(im2col(np.arange(9.0).reshape(3, 3), 3).shape)
