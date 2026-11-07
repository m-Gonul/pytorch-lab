import numpy as np


def swap_phase(a, b):
    Fa, Fb = np.fft.fft2(a), np.fft.fft2(b)
    return np.real(np.fft.ifft2(np.abs(Fa) * np.exp(1j * np.angle(Fb))))


def spectrum_log(img):
    S = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(img))))
    return np.round(255 * (S - S.min()) / (S.max() - S.min())).astype(np.uint8)


def shift(img, dx, dy):
    M, N = img.shape
    u = np.fft.fftfreq(M)[:, None]
    v = np.fft.fftfreq(N)[None, :]
    return np.real(np.fft.ifft2(np.fft.fft2(img) * np.exp(-2j * np.pi * (u * dy + v * dx))))


from skimage import data
print(spectrum_log(data.camera()).max())
