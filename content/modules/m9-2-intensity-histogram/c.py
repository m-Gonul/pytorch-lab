import numpy as np


def equalize(img, variant="opencv"):
    ...


def negative(img):
    ...


def log_transform(img):
    ...


def gamma(img, g):
    ...


from skimage import data
img = data.moon()
print(equalize(img).min(), equalize(img).max(), gamma(img, 0.5).mean())
