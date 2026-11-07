import numpy as np
from collections import deque


def region_grow(img, seeds, T):
    ...


from skimage import data
print(np.bincount(region_grow(data.coins(), [(100, 100), (50, 300)], 20).ravel()))
