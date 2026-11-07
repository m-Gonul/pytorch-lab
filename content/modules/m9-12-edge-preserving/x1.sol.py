import numpy as np
import cv2


def ahe_sliding(img, w):
    r = w // 2
    P = cv2.copyMakeBorder(img, r, r, r, r, cv2.BORDER_REFLECT_101)
    win = np.lib.stride_tricks.sliding_window_view(P, (w, w))
    c = (win <= img[:, :, None, None]).sum((-2, -1)) / (w * w)
    return np.rint(255 * c).astype(np.uint8)


def noise_gain(img, out, region):
    y0, y1, x0, x1 = region
    return float(out[y0:y1, x0:x1].astype(float).std() / img[y0:y1, x0:x1].astype(float).std())


from skimage import data
print(ahe_sliding(data.camera()[:64, :64], 15).mean())
