import math
import numpy as np
import cv2
import torch
from skimage import data
from labcheck import close, max_loops, need, ensure, bench, fmt_time


def check(ns, R):
    src = ns["__source__"]
    R.test("döngü yok", lambda: max_loops(src, 0, func="gradient_field"))
    imgs = np.stack([cv2.resize(i, (128, 128)).astype(np.float32) for i in (data.camera(), data.moon(), data.coins())])

    def eq():
        M, th, h = need(ns, "gradient_field")(torch.tensor(imgs), 9)
        for i, im in enumerate(imgs):
            gx = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=3)
            m = cv2.magnitude(gx, gy)
            close(M[i].numpy(), m, atol=1e-2, what=f"büyüklük {i}")
            t = np.mod(np.arctan2(gy, gx), np.pi)
            b = np.minimum((t / np.pi * 9).astype(int), 8)
            ref = np.bincount(b.ravel(), weights=m.ravel(), minlength=9)
            close(h[i].numpy(), ref, rtol=1e-3, atol=1e-1, what=f"histogram {i}")
        ensure(bool((th >= 0).all() and (th < math.pi).all()), "θ ∈ [0, π)")
    R.test("büyüklük ve yönelim histogramları = cv2 Sobel ile", eq)

    def ramp():
        yy, xx = torch.meshgrid(torch.arange(64.0), torch.arange(64.0), indexing="ij")
        a = math.radians(35)
        img = (xx * math.cos(a) + yy * math.sin(a))[None]
        M, th, h = need(ns, "gradient_field")(img, 18)
        ensure(h[0].argmax().item() == 3, f"35°'lik rampanın enerjisi 3. kutuda (30–40°) olmalı: {h[0].argmax().item()}")
    R.test("bilinen yönlü rampa doğru kutuda", ramp)
    if "gradient_field" not in ns:
        return
    X = torch.rand(64, 256, 256)
    Xn = X.numpy()
    t_loop = bench(lambda: [cv2.magnitude(cv2.Sobel(Xn[i], cv2.CV_32F, 1, 0), cv2.Sobel(Xn[i], cv2.CV_32F, 0, 1)) for i in range(64)], repeat=3, number=1)
    t_vec = bench(lambda: ns["gradient_field"](X), repeat=3)
    R.table("64 × 256 × 256", ["yöntem", "süre"], [["cv2 döngüsü (yalnız büyüklük)", fmt_time(t_loop)], ["gradient_field (büyüklük + yön + histogram)", fmt_time(t_vec)]])
