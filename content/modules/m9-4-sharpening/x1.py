import math
import numpy as np
import cv2


def operators():
    ...


def orientation_error(name, angles):
    ...


print({k: v[0].shape for k, v in operators().items()})
print(orientation_error("sobel", [10, 30, 60]))
