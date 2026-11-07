import numpy as np
from skimage import data
from labcheck import need, ensure


def check(ns, R):
    rng = np.random.default_rng(0)
    bg = data.camera()[::4, ::4][:96, :128].astype(np.float64)
    first = bg.copy(); first[30:62, 40:80] = 250
    frames = np.stack([first] + [bg] * 300)
    frames = np.clip(frames + rng.normal(0, 4, frames.shape), 0, 255).astype(np.uint8)
    gm = np.zeros(bg.shape, bool); gm[30:62, 40:80] = True

    def t():
        rows = []
        hl = {}
        for phi in (4, 16, 64):
            c = need(ns, "ghost_curve")(frames, gm, phi)
            ensure(len(c) == 300 and c[0] > 0.8, "ilk karelerde hayalet bölgesi ön plan olmalı")
            hl[phi] = need(ns, "half_life")(c)
            rows.append([phi, f"{c[0]:.2f}", f"{c[50]:.2f}", f"{c[150]:.2f}", f"{c[-1]:.2f}", hl[phi]])
        ensure(hl[4] < hl[16] < hl[64] or hl[64] == 300, f"φ büyüdükçe hayalet daha yavaş sönmeli: {hl}")
        R.table("Hayalet bölgesinde ön plan oranı (40 × 32 piksellik nesne 1. karede gidiyor)", ["φ", "kare 1", "kare 51", "kare 151", "kare 300", "yarılanma"], rows,
                note="Yayılım hayaleti kenarlardan içeri doğru emer: süre yaklaşık φ × (nesne yarı genişliği) ile ölçeklenir.")
    R.test("hayalet sönümlemesi", t)
