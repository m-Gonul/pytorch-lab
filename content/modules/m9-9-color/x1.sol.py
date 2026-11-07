import numpy as np


def intensity_slice(img, levels, colors):
    idx = np.searchsorted(np.asarray(levels), img, side="right")
    return np.asarray(colors, np.uint8)[idx]


def sin_pseudocolor(img, freq, phases):
    f = img.astype(np.float64)[..., None]
    ph = np.asarray(phases, np.float64)
    return np.round(255 * (1 + np.sin(2 * np.pi * freq * f / 255 + ph)) / 2).astype(np.uint8)


from skimage import data
print(intensity_slice(data.camera(), [64, 128, 192], [(0, 0, 0), (255, 0, 0), (0, 255, 0), (255, 255, 255)]).shape)
