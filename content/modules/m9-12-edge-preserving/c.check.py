import numpy as np
import cv2
from skimage import data
from labcheck import no_loops, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("döngü yok", lambda: no_loops(src))
    rows = []
    for name in ("camera", "moon", "coins", "page"):
        def t(name=name):
            im = getattr(data, name)()
            for clip, tiles in [(2.0, (8, 8)), (4.0, (4, 4)), (1.0, (16, 16)), (40.0, (8, 8)), (0.0, (8, 8)), (2.0, (7, 5))]:
                got = need(ns, "clahe_cv")(im, clip, tiles)
                want = cv2.createCLAHE(clip, tiles).apply(im)
                d = int((got != want).sum())
                ensure(d == 0, f"clip {clip}, tiles {tiles}: {d} piksel farklı")
            rows.append([name, f"{im.shape[1]} × {im.shape[0]}", "dolgu var" if im.shape[0] % 8 or im.shape[1] % 8 else "tam bölünür"])
        R.test(f"clahe_cv = cv2.createCLAHE ({name})", t)
    if rows:
        R.table("6 parametre kombinasyonunda birebir", ["görüntü", "boyut", "8 × 8 döşemede"], rows)
