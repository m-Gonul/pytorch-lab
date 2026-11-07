import numpy as np
import cv2


def local_enhance(img, E=4.0, k0=0.4, k1=0.02, k2=0.4, size=3):
    f = img.astype(np.float64)
    m = cv2.blur(f, (size, size), borderType=cv2.BORDER_REFLECT_101)
    s = np.sqrt(np.maximum(cv2.blur(f * f, (size, size), borderType=cv2.BORDER_REFLECT_101) - m * m, 0))
    mG, sG = f.mean(), f.std()
    sel = (m <= k0 * mG) & (s >= k1 * sG) & (s <= k2 * sG)
    return np.clip(np.where(sel, E * f, f), 0, 255).astype(np.uint8)


from skimage import data
print(local_enhance(data.camera()).mean())
