import math
import numpy as np
import torch


def slic(lab, K, c=10.0, iters=10):
    ...


from skimage import data, color
L = color.rgb2lab(data.astronaut()[:64, :64])
print(np.unique(slic(L, 16)).size)
