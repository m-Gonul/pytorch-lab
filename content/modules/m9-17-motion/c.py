import numpy as np
from scipy import ndimage


def block_match_np(prev, cur, bs, R):
    ...


def lk_flow(I0, I1, win=7, thr=1e-2):
    ...


from skimage import data
p = data.camera()[:64, :64]
print(block_match_np(p, np.roll(p, 1, 1), 8, 2)[2, 2])
