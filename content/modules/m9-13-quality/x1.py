import numpy as np
from scipy import ndimage


def ssim_map(x, y, win=7, sample_cov=True, R=255):
    ...


from skimage import data
print(ssim_map(data.camera(), data.camera())[0])
