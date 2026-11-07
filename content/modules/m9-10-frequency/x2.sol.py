import numpy as np


def _centered(img):
    return np.fft.fftshift(np.fft.fft2(img))


def find_spikes(img, k, r_dc=10):
    A = np.abs(_centered(img))
    M, N = A.shape
    u = np.arange(M)[:, None] - M // 2
    v = np.arange(N)[None, :] - N // 2
    A = np.where(u * u + v * v <= r_dc * r_dc, 0, A)
    idx = np.argsort(A.ravel())[::-1][:k]
    return [(int(i // N - M // 2), int(i % N - N // 2)) for i in idx]


def notch_reject(shape, centers, D0, n=2):
    M, N = shape
    u = np.arange(M)[:, None] - M // 2
    v = np.arange(N)[None, :] - N // 2
    H = np.ones(shape)
    for uk, vk in centers:
        Dk = np.sqrt((u - uk) ** 2 + (v - vk) ** 2)
        Dm = np.sqrt((u + uk) ** 2 + (v + vk) ** 2)
        with np.errstate(divide="ignore"):
            H *= 1 / (1 + (D0 / Dk) ** (2 * n)) * 1 / (1 + (D0 / Dm) ** (2 * n))
    return H


def remove_periodic(img, k, D0):
    sp = find_spikes(img, 2 * k)
    centers = []
    for p in sp:
        if (-p[0], -p[1]) not in centers:
            centers.append(p)
    H = notch_reject(img.shape, centers[:k], D0)
    return np.real(np.fft.ifft2(np.fft.ifftshift(H * _centered(img))))


print(notch_reject((32, 32), [(5, 3)], 2).min())
