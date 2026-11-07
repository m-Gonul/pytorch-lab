import numpy as np
import cv2


def otsu(img):
    p = np.bincount(img.ravel(), minlength=256) / img.size
    P1 = p.cumsum()
    mk = (np.arange(256) * p).cumsum()
    mg = mk[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sb = (mg * P1 - mk) ** 2 / (P1 * (1 - P1))
    return int(np.argmax(np.nan_to_num(sb, nan=0.0, posinf=0.0)))


def adaptive_mean(img, block, C):
    m = cv2.boxFilter(img, -1, (block, block), borderType=cv2.BORDER_REPLICATE)
    return np.where(img.astype(np.int32) > m.astype(np.int32) - C, 255, 0).astype(np.uint8)


from skimage import data
print(otsu(data.camera()))
