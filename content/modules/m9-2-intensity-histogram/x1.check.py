import numpy as np
from skimage import data, exposure
from labcheck import close, no_loops, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("döngü yok", lambda: no_loops(src))
    for a, b in (("moon", "camera"), ("camera", "coins"), ("page", "moon")):
        def t(a=a, b=b):
            s, r = getattr(data, a)(), getattr(data, b)()
            close(need(ns, "match_histograms")(s, r), exposure.match_histograms(s, r), atol=1e-9)
        R.test(f"{a} → {b} = skimage", t)
