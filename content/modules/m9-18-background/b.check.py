import numpy as np
from labcheck import close, no_imports, need, ensure


def check(ns, R):
    src = ns["__source__"]
    R.test("numpy kullanılmıyor", lambda: no_imports(src, {"numpy", "cv2", "torch"}))
    rng = np.random.default_rng(0)

    def vc():
        ensure(need(ns, "vibe_classify")([[[100]], [[130]], [[95]], [[180]], [[140]]], [[110]]) == [[0]], "elle örnek: arka plan")
        S = rng.integers(0, 256, (20, 12, 15)); f = rng.integers(0, 256, (12, 15))
        got = np.array(ns["vibe_classify"](S.tolist(), f.tolist(), 20, 2))
        want = ((np.abs(S - f) < 20).sum(0) < 2).astype(int)
        ensure(np.array_equal(got, want), "rastgele model")
    R.test("vibe_classify", vc)

    def ra():
        F = rng.integers(0, 256, (6, 4, 5)).astype(float)
        got = np.array(need(ns, "running_average")(F.tolist(), 0.1))
        B = F[0].copy(); want = [B.copy()]
        for f in F[1:]:
            B = 0.9 * B + 0.1 * f; want.append(B.copy())
        close(got, np.array(want), atol=1e-9, what="B_t")
        m = np.array(need(ns, "ema_masks")(F.tolist(), 0.1, 30))
        ensure(m.shape == (5, 4, 5) and np.array_equal(m, (np.abs(F[1:] - np.array(want)[:-1]) > 30).astype(int)), "maskeler önceki B'ye göre")
    R.test("running_average ve ema_masks", ra)
