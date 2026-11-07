import numpy as np


def warp_serialization(bins):
    ...


def cost(image, n_bins=256, private=False, block=256):
    ...


print(cost(np.zeros(1024, np.int64)), cost(np.arange(1024) % 256))
