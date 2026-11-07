import numpy as np
import cv2
from skimage import data
from labcheck import no_imports, need, ensure


def ref_clip(h, clip):
    h = np.array(h)
    ex = np.maximum(h - clip, 0).sum(); h = np.minimum(h, clip)
    n = len(h); h = h + ex // n; r = ex % n
    if r:
        step = max(n // r, 1); i = np.arange(n)
        h = h + ((i % step == 0) & (i // step < r))
    return h.tolist()


def check(ns, R):
    src = ns["__source__"]
    R.test("numpy/cv2 kullanılmıyor", lambda: no_imports(src, {"numpy", "cv2", "scipy", "skimage"}))

    def clip():
        ensure(need(ns, "clip_histogram")([10, 2, 0, 0, 0, 0, 0, 4], 4) == [5, 3, 1, 1, 1, 1, 0, 4], "elle örnek")
        rng = np.random.default_rng(0)
        for _ in range(50):
            h = rng.integers(0, 60, 256).tolist(); c = int(rng.integers(1, 40))
            got = ns["clip_histogram"](h, c)
            ensure(got == ref_clip(h, c), "rastgele histogram")
            ensure(sum(got) == sum(h), "toplam korunmalı")
    R.test("clip_histogram (OpenCV kuralı)", clip)

    def lut():
        tile = data.camera()[:64, :64]
        h = np.bincount(tile.ravel(), minlength=256).tolist()
        got = need(ns, "tile_lut")(h, 2.0, 4096)
        want = cv2.createCLAHE(2.0, (8, 8)).apply(data.camera())
        ensure(len(got) == 256 and all(0 <= v <= 255 for v in got) and got == sorted(got), "monoton 256'lık LUT")
        ensure(got[255] == 255, "LUT[255] = 255")
        raw = ns["tile_lut"](h, 0, 4096)
        slope = lambda L: max(b - a for a, b in zip(L, L[1:]))
        ensure(slope(got) <= slope(raw), "kırpma eğimi artıramaz")
        return f"LUT'un en büyük eğimi: kırpmasız {slope(raw)}, clip 2.0 ile {slope(got)}"
    R.test("tile_lut", lut)

    def bil():
        rng = np.random.default_rng(1)
        img = np.clip(data.camera()[200:240, 220:260] + rng.normal(0, 15, (40, 40)), 0, 255).astype(np.uint8)
        for d, sc, ss in [(7, 30, 5), (9, 50, 3)]:
            got = np.array(need(ns, "bilateral")(img.tolist(), d, sc, ss))
            want = cv2.bilateralFilter(img, d, sc, ss)
            diff = np.abs(got - want.astype(int))
            ensure(diff.max() <= 1 and (diff > 0).mean() < 0.01, f"d = {d}: en büyük fark {diff.max()}, farklı piksel %{(diff > 0).mean() * 100:.1f}")
    R.test("bilateral ≈ cv2.bilateralFilter (±1, float32 yuvarlama)", bil)
