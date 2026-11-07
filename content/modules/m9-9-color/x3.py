import numpy as np


def srgb_to_linear(c):
    ...


def rgb_to_xyz(img):
    ...


def xyz_to_lab(xyz):
    ...


def chromaticity(xyz):
    ...


def delta_e76(lab1, lab2):
    ...


print(xyz_to_lab(rgb_to_xyz(np.array([[[1.0, 1.0, 1.0]]]))))
