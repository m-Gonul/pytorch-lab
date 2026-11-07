import numpy as np


def pad(img, r, mode):
    ...


a = np.arange(12).reshape(3, 4)
print(pad(a, 2, "reflect101"))
