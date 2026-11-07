import numpy as np
import cv2
import torch
from skimage import data
from labcheck import max_loops, need, ensure, bench, fmt_time


def ref_vm(img, k):
    H, W, _ = img.shape
    r = k // 2
    P = np.pad(img.astype(np.float64), ((r, r), (r, r), (0, 0)), mode="edge")
    out = np.zeros_like(img)
    for y in range(H):
        for x in range(W):
            win = P[y:y + k, x:x + k].reshape(-1, 3)
            d = np.sqrt(((win[:, None] - win[None]) ** 2).sum(-1)).sum(1)
            out[y, x] = win[np.argmin(d)]
    return out


def noisy(seed, H=40, W=48):
    rng = np.random.default_rng(seed)
    im = data.astronaut()[200:200 + H, 300:300 + W].copy()
    m = rng.random((H, W)) < 0.15
    im[m] = rng.integers(0, 256, (m.sum(), 3))
    return im


def check(ns, R):
    src = ns["__source__"]
    R.test("en fazla bir döngü (bloklar)", lambda: max_loops(src, 1, func="vector_median"))
    for k in (3, 5):
        def t(k=k):
            im = noisy(k)
            got = need(ns, "vector_median")(torch.tensor(im), k).numpy()
            want = ref_vm(im, k)
            diff = (got != want).any(-1)
            ensure(diff.mean() < 0.002, f"{int(diff.sum())} piksel başvurudan farklı")
        R.test(f"vector_median = başvuru ({k} × {k})", t)

    def novel():
        im = noisy(7, 64, 64)
        vm = ns["vector_median"](torch.tensor(im), 3).numpy() if "vector_median" in ns else None
        mm = cv2.medianBlur(im, 3)
        P = np.pad(im, ((1, 1), (1, 1), (0, 0)), mode="edge")
        win = np.lib.stride_tricks.sliding_window_view(P, (3, 3), (0, 1)).transpose(0, 1, 3, 4, 2).reshape(64, 64, 9, 3)
        inwin = lambda out: (win == out[:, :, None, :]).all(-1).any(-1)
        ensure(vm is not None and inwin(vm).all(), "vektör median çıkışı penceredeki renklerden biri olmalı")
        clean = data.astronaut()[200:264, 300:364].astype(float)
        e = lambda o: np.abs(o.astype(float) - clean).mean()
        R.table("15% rastgele renk gürültüsü, 3 × 3", ["filtre", "penceredeki bir renk mi?", "ortalama mutlak hata"],
                [["marjinal median (cv2, kanal kanal)", f"%{inwin(mm).mean() * 100:.0f} piksel", f"{e(mm):.1f}"], ["vektör median", "%100 piksel", f"{e(vm):.1f}"]],
                note="Marjinal median kanalları ayrı sıraladığı için pencerede olmayan renkler üretebilir.")
    R.test("vektör median yeni renk üretmez", novel)
    if "vector_median" in ns:
        im = noisy(9, 64, 64)
        t1 = bench(lambda: ref_vm(im, 3), repeat=1, number=1)
        t2 = bench(lambda: ns["vector_median"](torch.tensor(im), 3), repeat=3)
        R.table("64 × 64, 3 × 3", ["yöntem", "süre"], [["piksel döngüsü (NumPy)", fmt_time(t1)], ["unfold + cdist", fmt_time(t2)]])
