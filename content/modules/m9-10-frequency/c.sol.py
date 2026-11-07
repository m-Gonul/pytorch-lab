import numpy as np


def lowpass(kind, shape, D0, n=2):
    P, Q = shape
    u = np.arange(P)[:, None] - P // 2
    v = np.arange(Q)[None, :] - Q // 2
    D = np.sqrt(u * u + v * v)
    if kind == "ideal":
        return (D <= D0).astype(np.float64)
    if kind == "butterworth":
        return 1 / (1 + (D / D0) ** (2 * n))
    if kind == "gaussian":
        return np.exp(-D ** 2 / (2 * D0 ** 2))
    raise ValueError(kind)


def highpass(kind, shape, D0, n=2):
    return 1 - lowpass(kind, shape, D0, n)


def freq_filter(img, H):
    M, N = img.shape
    P, Q = H.shape
    fp = np.zeros((P, Q))
    fp[:M, :N] = img
    sign = (-1.0) ** (np.arange(P)[:, None] + np.arange(Q)[None, :])
    G = H * np.fft.fft2(fp * sign)
    g = np.real(np.fft.ifft2(G)) * sign
    return g[:M, :N]


def conv_fft(img, k):
    s = (img.shape[0] + k.shape[0] - 1, img.shape[1] + k.shape[1] - 1)
    return np.fft.irfft2(np.fft.rfft2(img, s) * np.fft.rfft2(k, s), s)


from skimage import data
img = data.camera()
print(freq_filter(img, lowpass("gaussian", (1024, 1024), 30)).shape)
