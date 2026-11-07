import math
import numpy as np
import cv2


def operators():
    rob = (np.array([[-1, 0], [0, 1]], np.float32), np.array([[0, -1], [1, 0]], np.float32))
    pre = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], np.float32)
    sob = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    sch = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], np.float32)
    return {"roberts": rob, "prewitt": (pre, pre.T.copy()), "sobel": (sob, sob.T.copy()), "scharr": (sch, sch.T.copy())}


def orientation_error(name, angles):
    kx, ky = operators()[name]
    yy, xx = np.mgrid[0:21, 0:21].astype(np.float32)
    errs = []
    for a in angles:
        r = math.radians(a)
        img = np.sin(0.35 * (xx * math.cos(r) + yy * math.sin(r))).astype(np.float32) * 100
        gx = cv2.filter2D(img, -1, kx)[10, 10]
        gy = cv2.filter2D(img, -1, ky)[10, 10]
        est = math.degrees(math.atan2(gy, gx))
        if name == "roberts":
            est -= 45
        d = (est - a + 90) % 180 - 90
        errs.append(abs(d))
    return errs


print({k: v[0].shape for k, v in operators().items()})
print(orientation_error("sobel", [10, 30, 60]))
