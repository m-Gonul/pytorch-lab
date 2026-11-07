import numpy as np
from skimage.segmentation import slic
from sklearn.cluster import KMeans


def superpixel_kmeans(img, n_segments, k, seed=0):
    ...


def pixel_kmeans(img, k, seed=0):
    ...


from skimage import data
lab, sp = superpixel_kmeans(data.coffee()[::2, ::2], 300, 4)
print(lab.max(), sp.max())
