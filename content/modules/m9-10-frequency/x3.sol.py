import numpy as np


def dct_matrix(N):
    k = np.arange(N)[:, None]
    n = np.arange(N)[None, :]
    C = np.sqrt(2 / N) * np.cos(np.pi * (2 * n + 1) * k / (2 * N))
    C[0] /= np.sqrt(2)
    return C


def dct2(X):
    A, B = dct_matrix(X.shape[0]), dct_matrix(X.shape[1])
    return A @ X @ B.T


def idct2(Y):
    A, B = dct_matrix(Y.shape[0]), dct_matrix(Y.shape[1])
    return A.T @ Y @ B


def keep_largest(img, frac, kind):
    f = img.astype(np.float64)
    T = dct2(f) if kind == "dct" else np.fft.fft2(f)
    k = max(1, int(round(frac * T.size)))
    thr = np.sort(np.abs(T).ravel())[-k]
    T = np.where(np.abs(T) >= thr, T, 0)
    return idct2(T) if kind == "dct" else np.real(np.fft.ifft2(T))


print(np.round(dct_matrix(4) @ dct_matrix(4).T, 6))
