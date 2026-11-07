import numpy as np
import cv2


def _cw_median(img, k, w):
    r = k // 2
    win = np.lib.stride_tricks.sliding_window_view(np.pad(img, r, mode="edge"), (k, k)).reshape(*img.shape, k * k)
    extra = np.repeat(img[..., None], w - 1, -1)
    return np.median(np.concatenate([win, extra], -1), -1).astype(np.uint8)


def denoise_table(clean, noisy):
    return {
        "box5": cv2.PSNR(clean, cv2.blur(noisy, (5, 5))),
        "gauss7": cv2.PSNR(clean, cv2.GaussianBlur(noisy, (7, 7), 0)),
        "median5": cv2.PSNR(clean, cv2.medianBlur(noisy, 5)),
        "cwmedian5": cv2.PSNR(clean, _cw_median(noisy, 5, 3)),
    }


def fool_psnr(img):
    return np.clip(img.astype(np.int16) + 18, 0, 255).astype(np.uint8)


from skimage import data
print(fool_psnr(data.camera()).shape)
