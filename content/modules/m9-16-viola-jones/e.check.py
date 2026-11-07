import time
import numpy as np
import torch
from skimage import data
from labcheck import need, ensure, fmt_time


def scene(X, faces_idx, bg_idx, seed=0):
    rng = np.random.default_rng(seed)
    img = np.zeros((200, 200))
    cells = [(r, c) for r in range(8) for c in range(8)]
    for (r, c) in cells:
        p = X[rng.choice(bg_idx)]
        img[r * 25:(r + 1) * 25, c * 25:(c + 1) * 25] = p[:, ::-1] if rng.random() < 0.5 else p
    truth = []
    for i, (r, c) in zip(faces_idx, [(1, 1), (2, 5), (5, 2), (6, 6)]):
        img[r * 25:(r + 1) * 25, c * 25:(c + 1) * 25] = X[i]
        truth.append((c * 25, r * 25))
    return img, truth


def check(ns, R):
    X = data.lfw_subset()
    y = np.r_[np.ones(100), np.zeros(100)]
    perm = np.random.default_rng(0).permutation(200); tr, te = perm[:150], perm[150:]
    Rt, St, At = ns["feature_table"](3000, 0)
    F = torch.tensor(ns["normalized_features"](X, Rt, St, At))
    H = ns["train"](F[tr], torch.tensor(y[tr]), 30)

    def mc():
        cas = need(ns, "make_cascade")(F[tr][y[tr] == 1], H)
        ensure([k for k, _ in cas] == [2, 10, 30], "aşama boyutları")
        for k, thr in cas:
            s = sum(a * ((p * F[tr][y[tr] == 1][:, j]) <= (p * t)).double() for j, t, p, a in H[:k])
            ensure(bool((s >= thr).all()), f"aşama {k}: tüm eğitim yüzleri geçmeli")
    R.test("make_cascade: eğitim yüzlerinin hepsi geçer", mc)

    def sc():
        cas = ns["make_cascade"](F[tr][y[tr] == 1], H)
        img, truth = scene(X, [i for i in te if y[i] == 1][:4], [i for i in te if y[i] == 0])
        t0 = time.perf_counter()
        boxes, counts = need(ns, "scan")(img, Rt, St, At, H, cas)
        t1 = time.perf_counter()
        dets = need(ns, "nms")(boxes)
        found = sum(any(abs(x - tx) <= 3 and abs(y_ - ty) <= 3 for x, y_, _ in dets) for tx, ty in truth)
        ensure(counts[0] == 176 * 176, f"ilk aşamaya {counts[0]} pencere girmeli (176²)")
        ensure(counts[1] < counts[0] * 0.6, "ilk aşama pencerelerin çoğunu elemeli")
        ensure(found >= 3, f"4 yüzden {found} tanesi bulundu")
        evals = sum(c * k for c, (k, _) in zip(counts, cas))
        R.table("200 × 200 sahne (8 × 8 yama, 4'ü görülmemiş yüz)", ["aşama", "öznitelik", "giren pencere"], [[i + 1, k, c] for i, ((k, _), c) in enumerate(zip(cas, counts))],
                note=f"Bulunan yüz {found}/4, NMS sonrası toplam tespit {len(dets)}. Öznitelik değerlendirmesi: kaskadla {evals:,}, tek aşamalı 30 öznitelikle {counts[0] * 30:,} ({counts[0] * 30 / evals:.1f}× tasarruf). Süre {fmt_time(t1 - t0)}.")
    R.test("tarama + kaskad + NMS", sc)
