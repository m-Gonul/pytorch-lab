import numpy as np


def match_histograms(src, ref):
    ...


from skimage import data
print(match_histograms(data.moon(), data.camera()).mean(), data.camera().mean())
