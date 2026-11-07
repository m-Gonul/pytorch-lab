import numpy as np
from scipy import ndimage, signal


def gaussian_size(sigma):
    ...


def log_formula(sigma):
    ...


def log_via_laplacian(sigma):
    ...


def derivative_noise(noise_var, smooth=None, seed=0):
    ...


print(gaussian_size(1.4), log_formula(1.4).sum())
