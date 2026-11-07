import numpy as np
from skimage import data
from labcheck import close, need, ensure


def check(ns, R):
    cases = [("gaussian", dict(mean=5.0, sigma=12.0)), ("rayleigh", dict(a=10.0, b=400.0)), ("erlang", dict(a=0.2, b=4)),
             ("exponential", dict(a=0.1)), ("uniform", dict(a=-20.0, b=30.0))]
    for kind, p in cases:
        def t(kind=kind, p=p):
            z = need(ns, "make_noise")(kind, (200000,), np.random.default_rng(1), **p)
            ensure(z.shape == (200000,), "şekil")
            est = need(ns, "estimate")(kind, z)
            for k, v in p.items():
                ensure(abs(est[k] - v) < 0.03 * abs(v) + 0.3, f"{k}: {est[k]:.3f}, gerçek {v}")
            return ", ".join(f"{k} = {est[k]:.3f}" for k in p)
        R.test(f"{kind}: üret ve kestir", t)

    def spt():
        img = data.camera()
        out = need(ns, "add_salt_pepper")(img, 0.05, 0.08, np.random.default_rng(2))
        ensure(abs((out == 255).mean() - (img == 255).mean() - 0.05) < 0.005 and abs((out == 0).mean() - (img == 0).mean() - 0.08) < 0.006, "oranlar")
        ensure(img.dtype == out.dtype and not np.shares_memory(img, out), "kopya, uint8")
    R.test("add_salt_pepper", spt)
