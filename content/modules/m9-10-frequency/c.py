import numpy as np


def lowpass(kind, shape, D0, n=2):
    ...


def highpass(kind, shape, D0, n=2):
    ...


def freq_filter(img, H):
    ...


def conv_fft(img, k):
    ...


from skimage import data
img = data.camera()
print(freq_filter(img, lowpass("gaussian", (1024, 1024), 30)).shape)
