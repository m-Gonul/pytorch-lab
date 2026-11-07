import numpy as np


def find_spikes(img, k, r_dc=10):
    ...


def notch_reject(shape, centers, D0, n=2):
    ...


def remove_periodic(img, k, D0):
    ...


print(notch_reject((32, 32), [(5, 3)], 2).min())
