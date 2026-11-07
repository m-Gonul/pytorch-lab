import math
import numpy as np
import cv2
import torch
from skimage import data, color
from labcheck import close, need, ensure, bench, fmt_time


def shaky(T=40, seed=0):
    rng = np.random.default_rng(seed)
    big = (color.rgb2gray(data.astronaut()) * 255).astype(np.float32)
    frames, true = [], []
    for t in range(T):
        sx, sy, sa = 2.0 * t, 10 * math.sin(t / 10), 0.0
        jx, jy, ja = rng.normal(0, 3), rng.normal(0, 3), rng.normal(0, math.radians(0.6))
        x, y, a = 80 + sx + jx, 120 + sy + jy, sa + ja
        M = cv2.getRotationMatrix2D((x + 160, y + 120), math.degrees(a), 1.0)
        M[:, 2] -= [x, y]
        frames.append(cv2.warpAffine(big, M, (320, 240), flags=cv2.INTER_LINEAR))
        true.append((jx, jy, ja))
    return np.stack(frames), np.array(true)


def jitter(frames, est):
    m = est(frames)
    traj = np.vstack([[0, 0, 0], np.cumsum(m, 0)])
    k = 5
    P = np.pad(traj, ((k, k), (0, 0)), mode="edge")
    sm = np.stack([P[i:i + 2 * k + 1].mean(0) for i in range(len(traj))])
    return np.abs(traj - sm)[k:-k, :2].mean()


def check(ns, R):
    frames, _ = shaky()

    def sm():
        traj = np.random.default_rng(1).normal(size=(30, 3)).cumsum(0)
        got = need(ns, "smooth")(traj, 4)
        P = np.pad(traj, ((4, 4), (0, 0)), mode="edge")
        close(got, np.stack([P[i:i + 9].mean(0) for i in range(30)]), atol=1e-12, what="hareketli ortalama")
    R.test("smooth (cumsum)", sm)

    def em():
        m = need(ns, "estimate_motion")(frames[:6])
        ensure(m.shape == (5, 3), "(T−1, 3)")
    R.test("estimate_motion", em)

    def st():
        out, corr = need(ns, "stabilize")(frames)
        ensure(out.shape == frames.shape, "şekil")
        crop = lambda f: f[:, 30:-30, 40:-40]
        j0 = jitter(crop(frames), ns["estimate_motion"])
        j1 = jitter(crop(out), ns["estimate_motion"])
        ensure(j1 < 0.4 * j0, f"titreşim {j0:.2f} → {j1:.2f} piksel")
        t1 = bench(lambda: ns["stabilize"](frames), repeat=1, number=1)
        H, W = frames.shape[1:]
        t2 = bench(lambda: [cv2.warpAffine(f, np.float32([[1, 0, 1], [0, 1, 1]]), (W, H)) for f in frames], repeat=3)
        R.table("40 kare, 320 × 240", ["ölçü", "değer"], [["girdi titreşimi (yörüngenin yumuşaktan sapması, px)", f"{j0:.2f}"], ["stabilize çıktı", f"{j1:.2f}"],
                                                         ["stabilize (kestirim + tek grid_sample)", fmt_time(t1)], ["yalnız 40 cv2.warpAffine", fmt_time(t2)]])
    R.test("stabilize", st)
