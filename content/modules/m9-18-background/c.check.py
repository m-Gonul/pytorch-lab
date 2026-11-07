import numpy as np
from skimage import data
from labcheck import close, max_loops, need, ensure


def video(T=60, seed=0, sigma=6):
    rng = np.random.default_rng(seed)
    bg = data.camera()[::3, ::3][:120, :160].astype(np.float64)
    obj = data.coins()[40:80, 40:80].astype(np.float64)
    frames, gts = [], []
    for t in range(T):
        f = bg.copy(); gt = np.zeros(bg.shape, bool)
        x = 2 * (t - 5)
        if t >= 5 and x + 30 < 160:
            f[40:70, x:x + 30] = obj[:30, :30]; gt[40:70, x:x + 30] = True
        y2 = 104 - 2 * (t - 8)
        if t >= 8 and 0 <= y2 and y2 + 16 <= 120:
            f[y2:y2 + 16, 120:136] = 230; gt[y2:y2 + 16, 120:136] = True
        frames.append(np.clip(f + rng.normal(0, sigma, f.shape), 0, 255).astype(np.uint8)); gts.append(gt)
    return np.stack(frames), np.stack(gts)


def f1(pred, gt):
    tp = (pred & gt).sum(); fp = (pred & ~gt).sum(); fn = (~pred & gt).sum()
    return 2 * tp / max(2 * tp + fp + fn, 1)


def check(ns, R):
    src = ns["__source__"]
    R.test("piksel döngüsü yok", lambda: (max_loops(src, 0, func="vibe_init"), max_loops(src, 0, func="vibe_step")))
    frames, gts = video()

    def init():
        S = need(ns, "vibe_init")(frames[0], 20, 0)
        ensure(S.shape == (20, 120, 160), "(N, H, W)")
        P = np.pad(frames[0].astype(int), 1, mode="edge")
        win = np.lib.stride_tricks.sliding_window_view(P, (3, 3)).reshape(120, 160, 9)
        ensure(all((win == S[k][..., None]).any(-1).all() for k in range(20)), "her örnek 3 × 3 komşuluktan gelmeli")
    R.test("vibe_init", init)

    def upd():
        rng = np.random.default_rng(5)
        f = np.full((60, 80), 120, np.uint8)
        S = np.full((20, 60, 80), 118, np.int16)
        changed = []
        for _ in range(40):
            before = S.copy()
            fg = need(ns, "vibe_step")(S, f, rng)
            ensure(not fg.any(), "her şey arka plan olmalı")
            changed.append((before != S).sum() / f.size)
        rate = float(np.mean(changed))
        ensure(abs(rate - 2 / 16) < 0.03, f"kare başına değişen örnek / piksel {rate:.3f}, ≈ 2/φ = 0.125 bekleniyordu")
        S2 = np.full((20, 60, 80), 118, np.int16)
        fg = ns["vibe_step"](S2, np.full((60, 80), 250, np.uint8), rng)
        ensure(fg.all() and (S2 == 118).all(), "ön plan pikselleri hiçbir şeyi güncellememeli")
        return f"güncelleme oranı {rate:.3f} (beklenen 0.125: kendi + komşu)"
    R.test("vibe_step güncelleme kuralları", upd)

    def run():
        m = need(ns, "vibe_run")(frames)
        sc = f1(m[10:], gts[10:])
        naive = np.abs(frames[10:].astype(int) - frames[0].astype(int)) > 25
        ensure(sc > 0.75, f"F1 {sc:.3f}")
        R.table("Sentetik video (σ = 6 gürültü, iki nesne)", ["yöntem", "F1 (kare 10–59)"], [["|I_t − I_0| > 25", f"{f1(naive, gts[10:]):.3f}"], ["ViBe", f"{sc:.3f}"]])
    R.test("vibe_run", run)
