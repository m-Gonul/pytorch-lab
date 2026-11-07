import numpy as np


def match_histograms(src, ref):
    sv, si, sc = np.unique(src.ravel(), return_inverse=True, return_counts=True)
    tv, tc = np.unique(ref.ravel(), return_counts=True)
    sq = np.cumsum(sc) / src.size
    tq = np.cumsum(tc) / ref.size
    return np.interp(sq, tq, tv)[si].reshape(src.shape)


from skimage import data
print(match_histograms(data.moon(), data.camera()).mean(), data.camera().mean())
