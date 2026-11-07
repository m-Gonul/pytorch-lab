import math


def mse(x, y):
    ...


def psnr(x, y, R=255):
    ...


def ssim_global(x, y, L=255, K1=0.01, K2=0.03):
    ...


def ssim_components(x, y, L=255, K1=0.01, K2=0.03):
    ...


print(ssim_global([1, 2, 3, 4], [2, 3, 4, 5]))
