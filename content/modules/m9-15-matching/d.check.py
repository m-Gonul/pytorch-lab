import math
import numpy as np
import cv2
import torch
from skimage import data
from labcheck import close, need, ensure


def make_video(T=60, seed=0, S=64, rot=2.0):
    rng = np.random.default_rng(seed)
    bg = cv2.resize(data.moon(), (320, 240)).astype(np.float32)
    obj = data.camera()[180:180 + S, 200:200 + S].astype(np.float32)
    frames, centers = [], []
    cx, cy, vx, vy = 80.0, 80.0, 2.0, 1.2
    for t in range(T):
        f = bg.copy()
        x0, y0 = int(round(cx - S / 2)), int(round(cy - S / 2))
        M = cv2.getRotationMatrix2D((S / 2, S / 2), t * rot, 1.0)
        f[y0:y0 + S, x0:x0 + S] = cv2.warpAffine(obj, M, (S, S), borderMode=cv2.BORDER_REFLECT)
        frames.append(np.clip(f + rng.normal(0, 8, f.shape), 0, 255).astype(np.float32))
        centers.append((x0 + S / 2, y0 + S / 2))
        cx += vx + rng.normal(0, 0.5); cy += vy + math.sin(t / 6) * 1.5
        if not 40 < cx < 280:
            vx = -vx
        if not 40 < cy < 200:
            vy = -vy
    return frames, centers


def check(ns, R):
    def pre():
        p = np.random.default_rng(1).random((32, 48)) * 255
        q = need(ns, "preprocess")(p).numpy()
        z = np.log(p + 1); z = (z - z.mean()) / (z.std() + 1e-5)
        close(q, z * np.outer(np.hanning(32), np.hanning(48)), atol=1e-12, what="ön işleme")
        g = torch.fft.ifft2(need(ns, "target")(32, 48)).real
        ensure(tuple(np.unravel_index(int(g.argmax()), g.shape)) == (16, 24), "Gauss tepe merkezde")
    R.test("preprocess ve target", pre)

    def track():
        fr, c = make_video()
        tr = need(ns, "Mosse", "sınıf")(fr[0], *c[0])
        errs, psrs = [], []
        for f, cc in zip(fr[1:], c[1:]):
            x, y, p = tr.update(f)
            errs.append(math.hypot(x - cc[0], y - cc[1])); psrs.append(p)
        tm = cv2.getRectSubPix(fr[0], (64, 64), c[0])
        e2 = []
        for f, cc in zip(fr[1:], c[1:]):
            r = cv2.matchTemplate(f, tm, cv2.TM_CCOEFF_NORMED); y, x = np.unravel_index(r.argmax(), r.shape)
            e2.append(math.hypot(x + 32 - cc[0], y + 32 - cc[1]))
        ensure(np.mean(errs) < 6 and max(errs) < 15, f"MOSSE ortalama hata {np.mean(errs):.1f}, en büyük {max(errs):.1f}")
        ensure(np.mean(e2) > 20, "sabit şablon dönen hedefte kaymalı")
        R.table("60 kare, hedef her karede 2° dönüyor, σ = 8 gürültü", ["yöntem", "ortalama merkez hatası", "en büyük"],
                [["sabit şablon NCC", f"{np.mean(e2):.1f}", f"{max(e2):.1f}"], ["MOSSE (η = 0.125)", f"{np.mean(errs):.1f}", f"{max(errs):.1f}"]],
                note=f"MOSSE PSR: ortalama {np.mean(psrs):.1f}, en düşük {min(psrs):.1f} (Bolme: PSR < 7 izleme kaybı işareti).")
    R.test("dönen hedefi izle", track)
