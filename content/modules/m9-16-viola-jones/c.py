import numpy as np


def rect_table(coords, kinds):
    ...


def haar_matrix(X, R, S):
    ...


def best_stump(f, y, w):
    ...


from skimage import data
from skimage.feature import haar_like_feature_coord
c, k = haar_like_feature_coord(25, 25, "type-2-x")
R, S = rect_table(c[:10], k[:10])
print(haar_matrix(data.lfw_subset()[:3], R, S).shape)
