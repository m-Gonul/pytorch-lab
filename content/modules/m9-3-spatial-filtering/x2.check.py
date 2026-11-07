import numpy as np
import cv2
from labcheck import close, need, ensure


def check(ns, R):
    cases = {
        "Gauss 7×7": cv2.getGaussianKernel(7, 0) @ cv2.getGaussianKernel(7, 0).T,
        "kutu 5×3": np.ones((5, 3)) / 15,
        "Sobel x": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float),
    }
    for name, k in cases.items():
        def t(k=k):
            r = need(ns, "separate")(k)
            ensure(r is not None, "ayrılabilir olmalı")
            close(np.outer(r[0], r[1]), k, atol=1e-9)
        R.test(f"{name} ayrılabilir", t)
    for name, k in {"Laplasyen": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], float), "rastgele": np.random.default_rng(0).normal(size=(3, 3))}.items():
        R.test(f"{name} ayrılamaz", (lambda k=k: ensure(need(ns, "separate")(k) is None, "None bekleniyordu")))
